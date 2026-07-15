# Two composite scientific SYCL miniapps

This directory contains two standalone C++17/SYCL benchmarks that replace the
SPH experiment with application pipelines assembled from familiar PolyBench
operator families. They use only SYCL buffers/accessors and expose separate
`CONFIG`, `DAG`, `MEMORY`, `TIMING`, `RESULT`, and `VERIFY` records.

## 1. Multi-system real-space DFT / CheFSI

Source: `chebsi_dft_ensemble_sycl_timer.cpp`

The application model is high-throughput materials screening. Each `system`
is an independent real-space Kohn-Sham-like calculation with its own grid,
orbital block, potential, density, and small dense subspace matrices. One SCF
cycle is:

```text
orbital state + potential
  -> degree-D Chebyshev Hamiltonian filter
  -> overlap partial/reduce --------------------------+
  -> H * filtered orbitals -> projected-H partial/reduce
  -> approximate orthogonalization/Rayleigh-Ritz transform
  -> orbital block rotation -> density -> potential mixing
  -> next SCF cycle
```

The real-space Hamiltonian uses a periodic seven-point stencil. Overlap and
projected-H construction are blocked Gram products, and orbital rotation is a
dense block multiply. The small subspace transform is a stable diagonal plus
off-diagonal correction; it preserves the workload/dataflow of CheFSI but is
not a production generalized eigensolver.

| Phase | Closest PolyBench family | Dominant behavior |
| --- | --- | --- |
| Hamiltonian/Chebyshev filter | `jacobi-3d`/`heat-3d` plus vector recurrence | 3-D stencil, repeated state chain |
| overlap `Psi^T Psi` | `syrk` | reduction into a band-by-band Gram matrix |
| projected `Psi^T H Psi` | `gemm`/`syr2k` | second blocked Gram product |
| subspace rotation `Psi*T` | `gemm` | grid-by-band dense block multiply |
| density and mixing | vector kernels | band reduction and pointwise update |

With the defaults, each system contributes `cheb_degree + 9 = 21` kernels per
SCF cycle. Eight independent systems give 168 kernels per wait window, while
each system retains a long true dependency chain. `mixed` creates resolution
classes for heterogeneous task costs; `uniform` removes that heterogeneity;
`single-large` is the data-parallel control case.

Build and run:

```bash
clang++ -O3 -std=c++17 -fsycl chebsi_dft_ensemble_sycl_timer.cpp \
  -o chebsi_dft_ensemble_sycl_timer

./chebsi_dft_ensemble_sycl_timer \
  --nx 96 --ny 96 --nz 96 --bands 32 \
  --systems 8 --classes 4 --scf-cycles 8 --cheb-degree 12 \
  --partials 128 --mode mixed
```

To increase native `run_sec`, raise `--scf-cycles` first because it increases
timed work without increasing resident memory. Raise `--cheb-degree` for a
longer stencil chain, `--systems` for more schedulable pipelines, `--bands`
for Gram/rotation cost, and grid extents for stencil and orbital-block cost.
Memory grows approximately with `systems * nx * ny * nz * bands`.

## 2. Multi-region PDE forecast + LETKF

Source: `letkf_pde_ensemble_sycl_timer.cpp`

The application model is an ensemble weather/ocean forecast divided into
independent local regions. Each ensemble member advances a four-field,
periodic shallow-water-like state. Synthetic height observations then drive
an ensemble-transform analysis:

```text
ensemble state
  -> F forecast stencil steps -> mean + anomalies
  -> observation-space Gram ---------> transform approximation --+
  -> innovation RHS -----------------> mean weights --------------+
  -> analysis block update -> next forecast/analysis cycle
```

The transform and weights are computed once per region and applied to all grid
points. This corresponds to a localized/domain-decomposed EnKF workload. The
small-matrix inverse-square-root is a diagonal/off-diagonal approximation,
not a production symmetric eigendecomposition; the expensive and schedulable
forecast, observation projection, and ensemble analysis paths remain intact.

| Phase | Closest PolyBench family | Dominant behavior |
| --- | --- | --- |
| shallow-water forecast | `jacobi-2d`/`adi` | multi-field 2-D stencil chain |
| mean and anomalies | reductions/vector kernels | ensemble-axis reduction |
| observation Gram | `syrk` | observation-by-ensemble Gram matrix |
| innovation RHS | `gemv`/`atax` | observation projection |
| transform/weights | small dense/vector kernels | fan-out after Gram/RHS |
| analysis update | `gemm` plus `gemv` | ensemble transform at every cell |

With defaults, each region contributes `forecast_steps + 6 = 10` kernels per
cycle. Eight regions therefore give 80 kernels per wait window. The Gram and
RHS kernels fan out after anomaly construction, as do transform and weights;
the analysis update is their true fan-in.

Build and run:

```bash
clang++ -O3 -std=c++17 -fsycl letkf_pde_ensemble_sycl_timer.cpp \
  -o letkf_pde_ensemble_sycl_timer

./letkf_pde_ensemble_sycl_timer \
  --nx 256 --ny 256 --ensemble 32 \
  --regions 8 --classes 4 --cycles 12 --forecast-steps 4 \
  --obs-stride 4 --mode mixed
```

To increase native `run_sec`, raise `--cycles` first. `--forecast-steps`
lengthens the stencil chain without adding memory; `--regions` widens the DAG;
`--ensemble` increases forecast storage and makes Gram/analysis substantially
more expensive; reducing `--obs-stride` increases observation Gram/RHS work.
Grid size affects every large field kernel. Memory grows approximately with
`regions * nx * ny * ensemble`.

## Timing and scheduler controls

`init_sec` covers deterministic host generation and buffer construction.
`run_sec` begins after queue creation and includes submission, device work, and
the configured `queue.wait()` boundaries. `host_sec` is final materialization
and validation. Thus initialization is not silently counted as application
kernel time.

`--window-scf` and `--window-cycles` select how many outer iterations the
runtime sees before a wait. The default value `1` gives repeated profiling and
scheduling epochs. Larger values expose a larger cross-iteration DAG but can
increase scheduling overhead. `--wait-each-kernel 1` is only a serialization
diagnostic. `--host-read-full 1` is useful for validation but intentionally
increases `host_sec`.

Both programs assign a distinct buffer identity to every system or region.
They submit phases across all independent pipelines before waiting, so a
buffer-DAG scheduler can choose whole-pipeline placement and preserve data
residency. A later Celerity port should use neighborhood mapping for the
stencils and explicit full/sliced mappings for Gram and ensemble transforms;
using one-to-one mapping for those global reductions would be incorrect.
