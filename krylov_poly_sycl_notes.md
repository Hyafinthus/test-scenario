# Krylov Polynomial SYCL Miniapp Notes

`krylov_poly_sycl_timer.cpp` is the runtime-friendly replacement for the
full GMRES-style stress benchmark.  It keeps the Krylov/Arnoldi motivation but
uses a 3mm-like, simple buffer DAG that matches the current daemon and handler
split boundary.

## Why This Version Exists

The full GMRES miniapp exposed many real solver features at once: reductions,
scalar `H` buffers, `read_write` AXPY/projection kernels, reused partial
buffers, and more than one hundred kernels per cycle.  That is useful as a
future stress test, but it currently spends too much time debugging runtime
correctness.

This version keeps the research-facing structure:

- large 2-D matrix-free stencil applications,
- read-mostly Krylov basis planes,
- independent branch kernels inside each wait batch,
- a basis-mixing kernel whose loop count grows with the Krylov stage,
- a final multi-vector fan-in over basis planes,
- one `queue.wait()` per restart cycle for profile-guided scheduling.

It removes the fragile runtime boundary cases:

- no reductions,
- no scalar dependency buffers,
- no `read_write` accessors,
- no atomics,
- no host scalar decisions inside a cycle.

All timed kernels use `read` inputs and `discard_write` outputs.

## Data Layout

All large data is stored as true 2-D SYCL buffers.

```text
x, b, work, candidate, residual : (nx, ny)
basis                           : ((m + 1) * nx, ny)
branches                        : (branches * nx, ny)
```

Each logical basis plane is a row-block 2-D subrange:

```text
Q_k = basis[k * nx : (k + 1) * nx, 0 : ny]
```

This is compatible with the current split implementation because writes are
row-block ranges along dimension 0.

## Full Mode DAG

For every cycle:

```text
for k = 0..m-1:
  Q_k -> stencil -> work

  Q_k, b -> branch_0
  Q_k, b -> branch_1
  Q_k, b -> branch_2

  Q_0..Q_k, work, b, branches -> candidate
  candidate -> Q_(k+1)

Q_0..Q_fanout -> x

if not last cycle:
  x -> stencil -> work
  b, work -> residual
  residual, b -> Q_0

queue.wait()
```

The branch kernels are logically independent from the main stencil path and
from each other.  The `basis_mix` kernel is the per-stage convergence point:
it waits for the stencil and all branches, then writes the next basis plane.

## Kernel Count

Let:

```text
B = branches
m = Krylov polynomial depth
```

Each stage has:

```text
1 stencil/smooth seed
B independent branches
1 basis_mix
1 publish candidate
= B + 3 kernels
```

Default `B=3`, `m=8`:

```text
basis cycle        = 8 * 6 = 48 kernels
full non-last cycle = 48 + projection 1 + residual seed chain 3 = 52 kernels
full last cycle     = 48 + projection 1 = 49 kernels
final sample        = 1 kernel
```

Default `cycles=2` gives about `102` timed kernels, compared with `338` in the
full GMRES-style version.

## Relationship to Krylov Methods

This is not production GMRES.  It is a Krylov polynomial / stencil-basis
miniapp:

```text
Q_(k+1) = mix(A Q_k, Q_0..Q_k, b, independent branches)
x       = sum_i c_i Q_i
```

That still represents the runtime-relevant part of Krylov methods: repeated
operator application, a growing basis, read-mostly basis fan-in, and restart
cycles.  It deliberately omits orthogonalization reductions until the runtime
is stable enough for that stress case.
