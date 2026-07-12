# Krylov / GMRES SYCL miniapp notes

`krylov_gmres_sycl_timer.cpp` is a standalone, buffer/accessor-only
restarted GMRES / Arnoldi-inspired benchmark. It intentionally keeps the
algorithmic decomposition of a real solver while replacing solver-framework
machinery with a matrix-free five-point 2-D operator.

## Source inspection and provenance

The implementation was primarily derived from the algorithmic structure in
the locally cloned Ginkgo and `base-gmres` sources, then cross-checked against
the other available solver implementations.

- `base-ginkgo/core/solver/gmres.cpp` supplies the restarted GMRES sequence:
  operator application, modified Gram-Schmidt, norm/normalization, then a
  restart correction. Its `orthogonalize_mgs` helper is the direct source for
  the repeated `h(i,k) = dot(q_i,w)` then `w -= h(i,k) q_i` shape.
- `base-ginkgo/reference/solver/gmres_kernels.cpp` supplies the meaningful
  multi-AXPY correction pattern, `correction[idx] = sum_i Q[i*N+idx] * y[i]`.
- `base-gmres/Krylov.c` provides a compact textbook Arnoldi loop. Its
  `cuda_GMRES.cu` was useful specifically as a staging reference for matvec,
  partial dot reduction, final reduction, vector update, norm, normalization,
  and final basis-times-coefficient projection.
- `base-petsc/src/ksp/ksp/impls/gmres/gmres.c` and `borthog.c`,
  `base-kokkos-kernels/sparse/impl/KokkosSparse_gmres_impl.hpp`, and
  `base-trilinos/packages/belos/src/BelosPseudoBlockGmresIter.hpp` were
  inspected to cross-check the restarted operator → orthogonalize → normalize
  → small-system/correction structure.

This is not a library port. It does not carry framework executors, CSR matrix
classes, MPI, stopping criteria, Givens rotations, or a production
least-squares solve. Instead, it uses a deterministic matrix-free stencil and
deterministic restart coefficients. That keeps the final application small,
portable, and directly observable by a buffer-DAG runtime while preserving the
important GMRES dataflow.

## Diagnosis of the supplied runs

The three completed native runs in `base-result.txt` are structurally
consistent with the benchmark: memory estimates match the allocations, all
timings complete, sparse results are finite, ten cycles cost substantially
more than two, and sparse host materialization remains short. Because those
runs used `verify=0` and only sparse samples, they are a smoke/performance
check rather than a numerical GMRES reference test.

`our-result.txt` stops after exactly 306 deferred command groups. For the
default full mode, one Arnoldi cycle has
`sum(k=0..7, 8 + 3*k) = 148` kernels; two cycles add four marker/projection
kernels, the inter-cycle restart adds five, and final sampling adds one:
`2*148 + 4 + 5 + 1 = 306`. No scheduler-send, resubmit, timing, or result line
follows. This shows submission registration completed but execution dispatch
never began.

The immediate cause was the final `queue.wait()`: this runtime calls
`handler::scheduleOffline()` only from `event::wait()`. Sending all 306 kernels
in one event window would expose a second failure because both POSIX queues use
fixed 8192-byte messages and the handler does not check `mq_send` before
blocking in `mq_receive`. The revised benchmark retains returned events
and flushes at most 24 kernels per wait window. Native results should therefore
be re-measured from the revised source before reporting scheduler speedups;
the old native timings remain useful only as a structural reference.

## Algorithm and DAG shape

`Q` is one flattened buffer with `(m + 1)` contiguous vectors:

```text
Q[k * N + idx]
H[k * (m + 1) + i] = h(i, k)
```

For each Arnoldi step, the default/full path submits the following commands
without any host scalar decision; bounded event waits only close scheduler
windows:

```text
q_k -> 5-point stencil -> w
  + (q_k, b) -> independent residual-monitor branch -> auxiliary

for i = 0..k:
  (q_i, w) -> dot stage 1 partials -> dot stage 2 h(i,k) -> w -= h(i,k) q_i

w -> norm stage 1 partials -> norm stage 2 h(k+1,k) -> q_(k+1) = w / h(k+1,k)

(h(m,m-1), q_m) -> restart token -> x += token * sum_i y_i q_i
```

The two reduction stages use one work-item per contiguous partial chunk and a
one-work-item final reduction. This avoids local-memory and `sycl::reduction`
requirements. Scalar results are indexed inside a whole-`H` `read_write`
accessor because the current distributed scheduler tracks buffer identity but
does not transmit accessor offsets.

The restart token is deliberate: a correction using `q_0..q_(m-1)` would not,
by itself, depend on the final `q_m` normalization. The token reads the final
norm and `q_m`, then is consumed by the correction as a near-one stabilization
factor. It establishes the full restart boundary on the device without a host
scalar read or `queue.wait()`.

`discard_write` is used where a kernel completely overwrites a physical buffer.
This is the portable native-SYCL no-initialization equivalent used by the local
native examples; it avoids relying on a nonportable no-init property spelling.

## Mapping to the three research layers

| Research layer | Miniapp mechanism | What a runtime can observe |
| --- | --- | --- |
| Compute-cost semantics | MGS performs `k + 1` dot/update pairs at iteration `k`; fan-in cost grows with `--fanout`. | Range metadata alone does not encode the growing number of full-vector passes or dynamic fan-in loop cost. |
| Scheduling decisions | A large stencil and projection coexist with serial stage-2 reductions, memory-bound AXPY/auxiliary work, and an auxiliary branch independent of `w` for each iteration. | The stencil/projection are candidates for splitting; scalar reductions and short dependency-heavy stages may not be. The auxiliary branch gives real inter-kernel parallelism in addition to intra-kernel data parallelism. |
| Data maintenance / visibility | `Q` is read-mostly after each vector is produced; restart projection logically consumes `fanout` basis vectors plus coefficients to update one `x`. | The projection is a GMRES-style multi-vector fan-in with a large read set and one write target, exposing placement/coherence and virtual-buffer maintenance costs. |

The current scheduler identifies dependencies only by the underlying buffer
pointer and its cross-rank staging path moves whole buffers. Therefore all `Q`
accessors cover the whole flattened basis and kernels apply explicit offsets;
partial `Q` writers use `read_write`, ensuring that a device receiving the
latest `Q` also receives all earlier basis vectors. A future Celerity port can
restore per-vector mapper ranges. A one-element write guard makes the current
basis-writing and sparse-sampling kernels ineligible for SNMD split; their
global range does not match the flattened `(m+1)N` storage range.

## Experiment modes

| Mode | Purpose | Celerity expectation later |
| --- | --- | --- |
| `full` | Restarted stencil Arnoldi, growing MGS, device-side restart, and projection. | Mixed workload: a good end-to-end comparison. |
| `operator-only` | First stencil from `Q[0]`, then alternating between the N-element `w` and residual workspaces; parity selects the first destination so the final result lands in `w` without an extra copy. | Friendly: every timed algorithm kernel is an operator apply and every output range matches the global range, so a 2-D neighborhood / `one_to_one` mapping or SNMD split has a fair implementation. |
| `orthogonalization-only` | Deterministic `w` seed followed by MGS dots, AXPYs, norms, and normalizations; no stencil or projection. | Unfriendly/diagnostic: many small dependency-heavy reductions and changing MGS depth. |
| `fanin-only` | Repeated `x += Q*y` correction using initialized basis vectors and coefficients. | Exposes fan-in and data-maintenance/coherence pressure rather than stencil halo handling. |
| `no-fanin` | Full stencil Arnoldi cycles while skipping the restart correction. | Separates operator/orthogonalization scheduling from final fan-in cost. |

The present source deliberately contains no Celerity include or mapper. A
future Celerity version should express the logical stencil in 2-D with an
appropriate neighborhood/one-to-one mapper, not force a one-to-one mapping
onto the reduction or fan-in stages.

## Timing and materialization behavior

The default path uses deterministic host initialization and retains the public
event returned by each submit. It calls `event::wait()` after at most 24
kernels, because this modified runtime invokes `scheduleOffline()` only from
that API. `queue.wait()` and `event::wait_and_throw()` bypass the offline
scheduler in the current implementation and must not be used as window
boundaries.

The 24-kernel default also fits the runtime's fixed 8192-byte POSIX message:
the handler and daemon currently serialize every kernel and accessor in a wait
window into one request/reply message. The original 306-kernel full-mode window
exceeded this capacity. Each new window still contains a meaningful multi-
kernel DAG rather than forcing one wait per command. `--window-kernels` accepts
1 through 24; `--wait-each-kernel 1` remains the debug-only window-size-one
case.

After the final event window completes, the host materializes only a
four-element device-produced sample buffer. Therefore `host_sec` remains
separate from `run_sec` and normally avoids an unintended full `Q` or `x`
readback.

`--host-read-full 1` instead computes a host checksum over the primary output
for the selected mode.

The scheduler represents buffer sizes and cross-rank byte products with signed
`int` fields. The source consequently rejects any single physical buffer above
`INT_MAX` bytes before submission. `--allow-unsafe-large-buffer 1` bypasses
that guard for a stock/native-only run, but such a case is not safe for the
current distributed runtime. In particular, a flattened Q can hit this limit
well before total GPU memory is exhausted.

All SYCL buffers are constructed over persistent host `std::vector` storage.
This is required by the current cross-rank staging implementation, which calls
the runtime memory object's `getUserPtr()` when copying through shared memory.
The startup memory line therefore reports `host_backing_bytes` in addition to
the approximate device-buffer footprint.

The host and device initialization paths use the same deterministic q0
normalization, so `--init-on-device` changes placement rather than numerical
input.

The defaults are `nx=2048`, `ny=2048`, `m=8`, `cycles=2`,
`partials=1024`, and `window-kernels=24`. Startup prints exact buffer-byte
estimates plus the largest-buffer scheduler compatibility result.
