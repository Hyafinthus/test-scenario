# Multi-fidelity reactive-transport ensemble miniapp

## Why this replaces the Krylov positive case

`reactive_transport_ensemble_sycl_timer.cpp` is a real operator-split
reactive-transport workload, not another renamed element-wise benchmark.  Each
ensemble member solves three transported species with:

- a Robertson stiff-chemistry half-step;
- fifth-order WENO finite-volume flux reconstruction in both spatial axes;
- a conservative transport update followed by the second chemistry half-step.

The members represent a multi-fidelity uncertainty-quantification ensemble.
They use successively coarser meshes, different reaction-rate scales, and
different chemistry substep counts.  This is the source of task-level cost
heterogeneity; the different grid sizes are not fake scheduler guards.

The Robertson equations and Jacobian follow the `StiffChemistry` benchmark in
`base-kokkos-kernels/benchmarks/ode/KokkosODE_BDF.cpp`.  Kokkos Kernels solves
many independent ODE systems in parallel and uses BDF/Newton machinery.  The
miniapp keeps that scientific structure but expands a standalone 3x3
Rosenbrock-Euler solve in each cell, so the final program has no Kokkos,
oneMKL, MPI, USM, or external solver dependency.

The WENO5 finite-volume reconstruction is implemented directly because none of
the inspected base libraries provides a small standalone SYCL WENO frontend.
It uses positive, spatially varying velocities and clamped (zero-gradient)
domain boundaries.  The latter makes a radius-3 Celerity neighborhood mapper
exact; periodic wraparound would require a less precise mapper or explicit
ghost cells and would confound the comparison.

## Kernel DAG

For member `e` in macro-step `t`, the full-mode DAG is:

```text
state[e,t]
    |
    v
R[e]  Robertson reaction(dt/2)
   / \
  v   v
X[e] Y[e]  WENO5 x/y flux divergence
   \ /
    v
F[e]  transport update + Robertson reaction(dt/2)
    |
    v
state[e,t+1]
```

Dependencies inside one wait window are:

- `R[e] -> X[e]`: RAW on three chemistry-output species buffers;
- `R[e] -> Y[e]`: RAW on the same three buffers;
- `R[e] -> F[e]`: RAW on those three transported input fields;
- `X[e] -> F[e]`: RAW on three x-flux-divergence buffers;
- `Y[e] -> F[e]`: RAW on three y-flux-divergence buffers.

There are no dependencies between different members.  There are no in-place
updates: all outputs use `discard_write` and a different buffer.  Consequently
there are no intra-window WAR or WAW edges.  State buffers ping-pong between
steps.  This is an algorithm/implementation choice that makes intermediate
versions and DAG edges explicit; it is not a restriction of the target
runtime's `read_write` splitting support.

For `E` members, a full window contains `4E` kernels, has maximum width `2E`,
and has a critical path of three kernel durations (`R`, one of `X/Y`, `F`).
With the default `E=6`, this is 24 coarse kernels and width 12.  This is much
smaller in kernel count and much wider in useful work than the failed Krylov
Poly window (roughly 92 mostly small kernels with repeated immediate fan-in).

## Synchronization boundaries

One `queue.wait()` occurs after each complete physical macro-step.  This is a
semantic time-integration boundary and a repeated scheduling window, not an
arbitrary kernel-count workaround:

- step 0 uses cold-start estimates;
- step 1 and later can use device- and split-specific profiles;
- the next step reads the previous step's completed state;
- no waits occur between `R`, `X/Y`, and `F` unless
  `--wait-each-kernel 1` is selected for debugging.

## Compute geometry and cost

All four kernel types launch a 2-D `range(rows, cols)`.

| Kernel | Work per cell | Main cost property |
|---|---:|---|
| `R[e]` | `chem_substeps + 2e` expanded 3x3 linear solves | compute-heavy; member-dependent |
| `X[e]` | six WENO5 reconstructions (two interfaces x three species) | high-order radius-3 stencil |
| `Y[e]` | six WENO5 reconstructions (two interfaces x three species) | independent of `X[e]` |
| `F[e]` | nine-buffer transport fan-in plus the same chemistry half-step | compute-heavy producer fan-in |

With the default configuration, the six meshes contain 14,863,584 cells in
total and use 16, 18, 20, 22, 24, and 26 chemistry substeps respectively.  One
`R` layer therefore executes 290,147,904 cell-substeps, and the `R` plus `F`
halves execute 580,295,808 expanded Rosenbrock solves per macro-step.  Each
WENO direction performs 89,181,504 nonlinear reconstructions, or 178,363,008
across `X` and `Y`.  These are arithmetic-work counts rather than measured
FLOP/s; real-device timing is still required to prove that a particular
configuration amortizes profiling and transfer overhead.

Work is uniform over cells within a kernel, which is favorable to Celerity's
range splitting.  Cost is not uniform across stages or ensemble members:
chemistry substep count and mesh resolution change by member, while WENO and
chemistry have very different arithmetic intensity.  An access mapper can
describe all their regions correctly but cannot express those compute costs.

The current runtime profile key distinguishes the four useful shapes as
follows:

- multi-fidelity member meshes have different global and buffer ranges;
- reaction has three reads plus three writes;
- WENO has four reads plus three writes (`X` and `Y` intentionally share a
  symmetric cost/profile shape);
- final fan-in has nine reads plus three writes.

Thus repeated windows can learn useful costs without adding fake buffers or
dependencies.  A future profile key should still include stable kernel
identity so same-geometry multi-physics kernels can never alias accidentally.

## Memory geometry

Each member owns 17 row-major 2-D `data_t` fields:

- two ping-pong states, three species each: 6;
- chemistry half-step output: 3;
- x flux divergences: 3;
- y flux divergences: 3;
- x/y velocity fields: 2.

Every accessor covers a complete member field.  `R` and `F` write three full
fields; `X` and `Y` each write three full flux fields.  WENO reads a radius-3
neighborhood in one axis.  Reaction and final fan-in use one-to-one reads.
Initialization is deterministic, and each buffer has a normal backing host
vector like the local `3mm_timer.cpp` style.

## Split safety

The legal split dimension is NDRange dimension 0 (rows).  Each derived row
extent is aligned to four, matching the runtime's current candidate split
degrees 2 and 4.

For every write accessor:

- kernel global size is `(rows, cols)`;
- write-buffer range is `(rows, cols)`;
- a dim-0 execution chunk writes exactly the corresponding complete row block;
- a row block is contiguous in row-major storage;
- no two chunks write the same element.

The split implementation may replicate read-only input fields, but that is a
profitability issue rather than a correctness issue.  The daemon should reject
splitting when replication plus three output merges costs more than the saved
compute time.  No column-oriented or offset/subrange write relies on behavior
outside the current split model.

### `read_write` split contract

The target system may also split a `read_write` kernel.  Its required semantic
contract is:

1. every split part receives the complete pre-kernel input version needed by
   its reads;
2. work-items are partitioned into disjoint owned row blocks;
3. each work-item writes only elements in its part's owned write block;
4. merge copies back only that owned block, rather than the complete private
   replica.

Under this contract, `read_write` is correct even though its read data is fully
replicated.  The important condition is **partition-disjoint writes**, not the
access mode by itself.  A generic runtime cannot derive the actual write set,
overlap behavior, snapshot requirements, or safe merge interval merely from a
SYCL `read_write` accessor, so this is a system/benchmark contract that must be
stated explicitly.

For this miniapp, an in-place version of the pointwise `R` stage would satisfy
the contract because each work-item reads and writes only its own cell.  A
pointwise final update could do so as well if its state input and output were
aliased deliberately.  In contrast, `X` or `Y` must not overwrite the same
field from which their neighboring work-items read WENO stencils: that would
require an explicit snapshot or double buffering and cannot be justified by
owned write ranges alone.  The shipped implementation keeps all four stages
out of place to expose clean producer versions and to make the comparison easy
to audit, not because the runtime forbids `read_write`.

## Scheduling opportunity

This DAG exposes joint inter- and intra-kernel choices that a 3mm graph does
not sustain:

- place medium/coarse members whole on different GPUs and keep their four
  kernels co-located;
- overlap `X[e]` and `Y[e]` when spare devices make the input replication
  worthwhile;
- split only the finest reaction/final kernels when their measured compute
  time amortizes input replication and three partition merges;
- avoid splitting coarse members whose kernels are already short;
- assign more work to a faster GPU in a heterogeneous 4090/A6000-style node;
- react to transient load using monitoring penalties without discarding HEFT
  available-time state.

The important decision is whether distributed execution is profitable for a
kernel in the context of the entire ensemble DAG, not device-subset selection
by itself.

## Data movement and visibility

At startup, fields are host initialized.  In a profitable steady state:

- a whole member's state and scratch data stay on its selected GPU across
  macro-step waits;
- `R` output is reused by both WENO branches and by `F`;
- `X/Y` flux outputs are consumed once by `F`;
- only a split finest-grid kernel needs read replication and output merge;
- independent members do not exchange data;
- final host access is timed separately from `run_sec`.

If the scheduler moves `X`, `Y`, and `F` independently without accounting for
their shared producer, the extra copies can erase the gain.  That outcome would
reject the Layer-B/Layer-C hypothesis rather than show that the workload has no
parallelism.

## Fair Celerity 0.6 mapping

The checked local Celerity source is `code-celerity` at tag `v0.6.0`.  A fair
Celerity port should use:

- reaction reads/writes: `access::one_to_one`;
- WENO-x species and x-velocity reads: `access::neighborhood(0, 3)`;
- WENO-y species and y-velocity reads: `access::neighborhood(3, 0)`;
- WENO outputs: `access::one_to_one`;
- final transport/reaction reads and writes: `access::one_to_one`.

These are accurate mappers.  No result should be attributed to mapper misuse.
Celerity's virtual buffers can keep only the local chunks backed, exchange
compact radius-3 halos for WENO, overlap independent instructions through its
out-of-order engine, and avoid whole read-buffer replication.  Those are real
advantages, especially when a member exceeds one device's memory.

In the inspected v0.6 implementation, distributed command generation divides
splittable tasks deterministically over nodes, and local instruction generation
coarsely splits a device task over the node's available devices.  Split
constraints and 1-D/2-D/oversubscription hints affect granularity, but they do
not perform a profile-guided global search over whether each member should use
one device, a subset, or every device.  Therefore the comparison hypothesis is:

- Celerity will be strong on each individual WENO task because its mapper and
  halo are regular;
- it may pay unnecessary per-task multi-device participation, halo/coherence,
  and instruction costs for coarse members;
- equal chunks can also straggle on heterogeneous GPUs;
- this runtime can instead exploit ensemble task parallelism, selectively
  split only sufficiently large kernels, and preserve whole-member locality.

This is a hypothesis, not a guaranteed result.  On homogeneous GPUs with only
one very large member, Celerity may match or beat this runtime.  The intended
positive case is several differently sized members on multiple, preferably
heterogeneous, GPUs.

## Three-layer evidence and rejection criteria

### Layer A: compute-cost semantics

Hypothesis: access ranges alone do not predict the stage/member duration;
profile EWMA converges to materially different `R`, `X/Y`, and `F` costs.

Measure per key/device/split duration over steps.  Reject the hypothesis if
durations are indistinguishable or profile overhead remains a large fraction
of every kernel.

### Layer B: partitioning and scheduling

Hypothesis: after the cold window, HEFT overlaps whole-member pipelines, keeps
coarse kernels single-device, and splits only profitable fine kernels.  The
resulting makespan is lower than native single-GPU SYCL and Celerity's
deterministic participation pattern.

Measure placement, split degree, device timelines, makespan, and GPU
utilization.  Reject the hypothesis if the scheduler collapses to one GPU, if
all kernels are split, or if predicted finish-time improvements do not appear
in wall time.

### Layer C: data maintenance

Hypothesis: member-local placement avoids most halo/copy/coherence work and
keeps state resident across waits; selective split movement is smaller than
the saved compute time.

Measure H2D/D2D/D2H bytes, split pre-copy and merge time, producer-consumer
device changes, and final host materialization separately.  Reject the
hypothesis if movement dominates or if waits force all fields back to host.

## Suggested runs

Small correctness/syntax run:

```bash
./reactive_transport_ensemble_sycl_timer \
  --nx 64 --ny 64 --members 3 --steps 2 --chem-substeps 2 --verify 1
```

Celerity-friendly transport control:

```bash
./reactive_transport_ensemble_sycl_timer \
  --nx 1024 --ny 1024 --members 6 --steps 8 --mode transport-only
```

Main local multi-GPU run:

```bash
./reactive_transport_ensemble_sycl_timer \
  --nx 2048 --ny 2048 --members 6 --steps 20 \
  --chem-substeps 16 --coarsen-percent 10 --mode full
```

Compile with the same modified DPC++ toolchain used by the runtime, for example:

```bash
icpx -O3 -std=c++17 -fsycl reactive_transport_ensemble_sycl_timer.cpp \
  -o reactive_transport_ensemble_sycl_timer
```

Use identical member shapes, precision, steps, chemistry substeps, and timed
host-materialization policy in native SYCL, modified-runtime, and Celerity
runs.  Report both `run_sec` and `host_sec`; do not hide Celerity's data-size
advantage or charge only one implementation for final materialization.
