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

## Algorithm and DAG shape

`Q` is one flattened buffer with `(m + 1)` contiguous vectors:

```text
Q[k * N + idx]
H[k * (m + 1) + i] = h(i, k)
```

For each Arnoldi step, the default/full path submits the following commands
without a host wait:

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
requirements, while small scalar `H` subranges carry device-side dependencies
to the AXPY and normalization kernels.

The restart token is deliberate: a correction using `q_0..q_(m-1)` would not,
by itself, depend on the final `q_m` normalization. The token reads the final
norm and `q_m`, then is consumed by the correction as a near-one stabilization
factor. It establishes the full restart boundary on the device without a host
scalar read or `queue.wait()`.

`discard_write` is used where a kernel completely overwrites an output range.
This is the portable native-SYCL no-initialization equivalent used by the local
native examples; it avoids relying on a nonportable no-init property spelling.

## Mapping to the three research layers

| Research layer | Miniapp mechanism | What a runtime can observe |
| --- | --- | --- |
| Compute-cost semantics | MGS performs `k + 1` dot/update pairs at iteration `k`; fan-in cost grows with `--fanout`. | Range metadata alone does not encode the growing number of full-vector passes or dynamic fan-in loop cost. |
| Scheduling decisions | A large stencil and projection coexist with serial stage-2 reductions, memory-bound AXPY/auxiliary work, and an auxiliary branch independent of `w` for each iteration. | The stencil/projection are candidates for splitting; scalar reductions and short dependency-heavy stages may not be. The auxiliary branch gives real inter-kernel parallelism in addition to intra-kernel data parallelism. |
| Data maintenance / visibility | `Q` is read-mostly after each vector is produced; restart projection reads a contiguous `fanout * N` basis range plus coefficients to update one `x`. | The projection is a GMRES-style multi-vector fan-in with a large read set and one write target, exposing placement/coherence and virtual-buffer maintenance costs. |

The program uses exact per-vector `Q` subranges and one-element `H` subranges
for the critical Arnoldi commands. Those ranges preserve useful dependency and
placement metadata instead of declaring the entire basis writable for each
small operation.

## Experiment modes

| Mode | Purpose | Celerity expectation later |
| --- | --- | --- |
| `full` | Restarted stencil Arnoldi, growing MGS, device-side restart, and projection. | Mixed workload: a good end-to-end comparison. |
| `operator-only` | Alternating regular stencil applications between `Q[0]` and `w`. | Friendly: a 2-D neighborhood / `one_to_one` mapping should give this phase a fair distributed implementation. |
| `orthogonalization-only` | Deterministic `w` seed followed by MGS dots, AXPYs, norms, and normalizations; no stencil or projection. | Unfriendly/diagnostic: many small dependency-heavy reductions and changing MGS depth. |
| `fanin-only` | Repeated `x += Q*y` correction using initialized basis vectors and coefficients. | Exposes fan-in and data-maintenance/coherence pressure rather than stencil halo handling. |
| `no-fanin` | Full stencil Arnoldi cycles while skipping the restart correction. | Separates operator/orthogonalization scheduling from final fan-in cost. |

The present source deliberately contains no Celerity include or mapper. A
future Celerity version should express the logical stencil in 2-D with an
appropriate neighborhood/one-to-one mapper, not force a one-to-one mapping
onto the reduction or fan-in stages.

## Timing and materialization behavior

The default path uses deterministic host initialization, submits the full
benchmark DAG, calls `queue.wait()` once at the final timing boundary, and
materializes only a four-element device-produced sample buffer. Therefore the
reported `host_sec` is separated from `run_sec` and normally avoids an
unintended full `Q` or `x` host readback.

`--host-read-full 1` instead computes a host checksum over the primary output
for the selected mode. `--wait-each-kernel 1` is intentionally a debug-only
comparison switch; it inserts a wait after every submitted command and removes
the wait-window opportunity that the default benchmark is intended to expose.

The defaults are `nx=2048`, `ny=2048`, `m=8`, `cycles=2`, and
`partials=1024`. Startup prints exact buffer-byte estimates so larger
H100-class cases can be chosen from available device memory rather than by
guessing.
