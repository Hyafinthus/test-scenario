You are working in my local HPC/SYCL runtime research workspace.

Goal:
Create one standalone SYCL buffer-accessor benchmark application derived from a real Krylov / GMRES / Arnoldi-style solver. The application must be suitable as the main motivating/evaluation application for comparing:
  1. Native single-process SYCL baseline,
  2. My modified distributed SYCL runtime,
  3. Later, a Celerity version.

I have already cloned these source repositories locally:
  - ginkgo/
  - petsc/
  - trilinos/
  - kokkos-kernels/
  - GPGPU-GMRES-Method/

I also have existing local benchmark examples:
  - 3mm_timer.cpp
  - test_copy.cpp

Use those two files as style references for:
  - SYCL buffer/accessor style,
  - command-line argument handling,
  - timing format,
  - minimal dependencies,
  - avoiding unnecessary waits,
  - using buffer-accessor semantics so that my runtime can intercept kernel DAGs before wait.

Do not rewrite a full external library. Instead:
  1. Inspect the available repositories to understand the real GMRES / Arnoldi / Krylov kernel structure.
  2. Choose the best source of algorithmic structure.
  3. Implement a standalone C++17/SYCL benchmark file in this workspace.

The final application should be a realistic Krylov/GMRES/Arnoldi-inspired miniapp, not just another PolyBench kernel.

================================================================================
Research context and required application properties
================================================================================

The application must expose three layers of application parallelization:

Layer 1: Compute-cost semantics
  - Access metadata / range information describes data needs, but not compute cost.
  - The application should contain kernels whose cost changes across iterations or depends on algorithmic stage.
  - Example: Arnoldi / modified Gram-Schmidt orthogonalization where the number of basis vectors grows with k.
  - The code should make this visible in the kernel structure.

Layer 2: Scheduling decisions
  - The application should contain both:
      a. kernels that are worth splitting over multiple devices,
      b. kernels that are small, sequential, reduction-heavy, or memory-bound and may not be worth splitting.
  - It should contain enough kernels and DAG structure so my runtime can benefit from wait-window DAG scheduling.
  - There should be inter-kernel parallelism and intra-kernel data parallelism.
  - Avoid putting queue.wait() after every kernel. Only wait at necessary algorithm boundaries or final timing boundaries.

Layer 3: Data-maintenance / data-visibility
  - The application should contain read-mostly basis vectors and multi-vector fan-in.
  - It should include a restart/projection/update phase like:
        x += sum_i y_i * q_i
    or a similar fan-in consumer reading many basis vectors and writing one output.
  - This phase should resemble the test_copy fan-in pattern but be algorithmically meaningful in GMRES / Arnoldi.
  - It should expose large read-only/read-mostly data and producer-consumer data placement opportunities.

Celerity should have known advantages in part of the application:
  - Include a large regular operator-apply phase, e.g. 2D/3D stencil or CSR SpMV, that would be Celerity-friendly with a neighborhood / one_to_one mapper.
  - Do not make Celerity look bad only because of wrong mapper.
  - The application should contain both Celerity-friendly and Celerity-unfriendly phases.

================================================================================
Implementation target
================================================================================

Create a standalone file, preferably:

  krylov_gmres_sycl_timer.cpp

If needed, create a small build script:

  build_krylov_gmres_sycl.sh

But do not introduce CMake unless necessary.

The program must compile with a generic SYCL compiler. Prefer code compatible with DPC++ / AdaptiveCpp-style SYCL where possible.

Use:
  - C++17
  - SYCL buffer/accessor model
  - No USM
  - No external library dependency in the final benchmark
  - No vendor-specific extensions
  - No oneMKL
  - No Eigen
  - No thrust
  - No Kokkos/Ginkgo/PETSc/Trilinos headers in the final code

You may inspect Ginkgo / PETSc / Trilinos / Kokkos Kernels only to understand algorithmic decomposition and naming. The final output should be a clean standalone benchmark.

================================================================================
Mandatory SYCL style constraints
================================================================================

Use SYCL buffers and accessors only.

Do NOT use:
  - malloc_device
  - malloc_shared
  - malloc_host
  - USM pointers
  - explicit cuda/hip calls
  - external BLAS
  - std::execution
  - C++ parallel algorithms

Use:
  - sycl::buffer<T, 1> and/or sycl::buffer<T, 2>
  - queue.submit(...)
  - handler.parallel_for(...)
  - accessor read_only / write_only / read_write
  - no_init where appropriate
  - deterministic initialization on host

Important:
  - Do not call q.wait() after every submit.
  - Build a meaningful DAG through buffer dependencies.
  - Use waits only:
      1. after initialization if absolutely needed,
      2. at restart-cycle or final timing boundaries,
      3. before reading final result on host,
      4. for correctness-critical scalar host decisions if impossible to avoid.
  - Prefer storing scalar values such as h[i,k], norms, dot products, and coefficients in small SYCL buffers so kernels depend through accessors instead of forcing host waits.

Kernel names:
  Use named kernel structs/classes or unique lambda names if needed by the compiler.

Timing:
  Follow the style of 3mm_timer.cpp and test_copy.cpp:
  - initialize,
  - run timed section,
  - q.wait() or equivalent final wait,
  - measure runtime,
  - read/sample result,
  - measure host access time separately if possible,
  - print a compact summary.

================================================================================
Required algorithm structure
================================================================================

Implement a restarted GMRES / Arnoldi-like miniapp.

Recommended structure:

Input:
  - A matrix-free 2D stencil operator or CSR SpMV.
  - Prefer matrix-free 2D stencil for simplicity and Celerity-friendliness:
        y[i,j] = 4*x[i,j] - x[i-1,j] - x[i+1,j] - x[i,j-1] - x[i,j+1]
    with boundary handling.
  - Store vectors as flattened 1D arrays of length N = nx * ny.
  - Store basis Q as m+1 vectors:
        Q[(m+1) * N]
    flattened as Q[k * N + idx].
  - Store H as small buffer:
        H[(m+1) * m]
  - Store temporary vector w[N], residual r[N], output x[N], b[N], etc.

Main flow:
  1. Initialize x, b, q0, basis Q, H, workspace buffers.
  2. For restart cycles:
      a. Apply operator:
             w = A * q_k
         This is the Celerity-friendly large regular kernel.
      b. Modified Gram-Schmidt / Arnoldi:
             for i = 0..k:
                h[i,k] = dot(q_i, w)
                w = w - h[i,k] * q_i
         Use device kernels for reductions and vector updates.
      c. Norm:
             h[k+1,k] = ||w||
             q_{k+1} = w / h[k+1,k]
      d. Optional block / multi-RHS or independent branch:
             include at least one independent vector operation branch per iteration if practical, so inter-kernel scheduling can be exposed.
      e. At restart end:
             x = x + sum_i y_i * q_i
         This is the multi-vector fan-in phase.
         y_i can be generated deterministically on host before the timed section or computed by a simple device kernel. It does not need to solve an exact least-squares system. The important part is the fan-in over many basis vectors.

Correctness:
  - The benchmark does not need to be a production-quality GMRES solver.
  - It should be algorithmically meaningful and deterministic.
  - It must produce a final checksum/sample so the compiler cannot remove work.
  - Use simple deterministic initialization.
  - Include optional correctness sanity checks if cheap.

Reduction implementation:
  - Avoid host synchronization for every dot if possible.
  - Implement a two-level reduction:
      kernel reduce_stage1:
        each work-group or fixed chunk writes partial sums to partial buffer.
      kernel reduce_stage2:
        reduces partials to a scalar buffer H[i,k] or norm buffer.
  - Then vector update kernel reads H[i,k] from small buffer.
  - If the compiler/runtime has issues with local memory, keep the reduction simple and portable, e.g. one work-item per chunk serially accumulates a block into partials. It may be slower but acceptable for benchmark structure.
  - Make the number of partial chunks configurable.

Important:
  - Do not use sycl::reduction if it complicates portability with my runtime. Prefer explicit partial buffers.

================================================================================
Parameterization
================================================================================

The program should accept command-line arguments, with sensible defaults:

  --nx <int>              default 4096 or 8192 if memory allows
  --ny <int>              default 4096 or 8192 if memory allows
  --m <int>               Krylov basis size / restart length, default 8 or 12
  --cycles <int>          restart cycles, default 2 or 3
  --partials <int>        number of reduction partials, default 1024 or 4096
  --fanout <int>          optional number of vectors used in final fan-in; default m
  --verify <0/1>          default 0
  --print-samples <0/1>   default 1

Memory must be estimated and printed at startup:
  - N = nx * ny
  - vector bytes
  - Q bytes
  - total approximate buffer bytes

Avoid default parameters that exceed a 16GB GPU. Provide comments for large H100-class runs.

Suggested default for safe run:
  nx=2048, ny=2048, m=8, cycles=2

Suggested larger runs:
  nx=8192, ny=8192, m=8 or 12
  nx=16384, ny=16384, m=4 or 8 if memory allows

================================================================================
Feature switches for my runtime / experiments
================================================================================

Add compile-time or runtime switches that allow controlled experiments:

Runtime flags preferred:
  --mode full
  --mode no-fanin
  --mode fanin-only
  --mode operator-only
  --mode orthogonalization-only

Meaning:
  full:
    Run the whole miniapp.
  operator-only:
    Run only repeated stencil/operator apply kernels.
    This should be Celerity-friendly.
  orthogonalization-only:
    Run basis dot/update/norm kernels.
    This should expose many-small-kernel / changing-cost behavior.
  fanin-only:
    Initialize Q and y, then run only x += sum_i y_i*q_i.
    This should resemble test_copy large fan-in but with GMRES basis vectors.
  no-fanin:
    Run full Arnoldi loop but skip final projection fan-in.

Also add:
  --wait-each-kernel <0/1> default 0
    If 1, force q.wait() after every kernel for debugging only.
    If 0, avoid waits except final/required boundaries.

  --host-read-full <0/1> default 0
    If 0, sample only sparse elements from x/Q.
    If 1, read full output checksum on host.
    This helps compare host materialization behavior.

  --init-on-device <0/1> default 0
    Optional. If easy, initialize buffers using kernels to reduce host transfer.
    If not easy, keep host initialization.

================================================================================
Output format
================================================================================

Print machine-readable lines like:

  CONFIG nx=... ny=... N=... m=... cycles=... partials=... mode=...
  MEMORY vector_bytes=... Q_bytes=... total_estimated_bytes=...
  TIMING init_sec=...
  TIMING run_sec=...
  TIMING host_sec=...
  RESULT checksum=... sample0=... sample1=... sample2=...

If implementing separate timers for phases, print:

  PHASE operator_sec=...
  PHASE orthogonalization_sec=...
  PHASE fanin_sec=...

But do not insert extra waits just to time every kernel unless --wait-each-kernel=1 or mode isolates the phase. Phase timing can be coarse.

================================================================================
Coding details
================================================================================

Use a single precision type switch:

  using data_t = float;

Optionally support double via:

  #ifdef USE_DOUBLE
  using data_t = double;
  #else
  using data_t = float;
  #endif

Avoid iostream-heavy code inside kernels. Host code can use iostream or printf.

Use flattened indexing:

  idx = i * ny + j

For Q:

  Q[k * N + idx]

For H:

  H[i * m + k] or H[k * (m + 1) + i], but document layout.

Kernel functions to implement:
  - init_vectors_kernel if using device init
  - stencil_apply_kernel
  - vector_copy_or_scale_kernel
  - dot_stage1_kernel
  - dot_stage2_kernel
  - axpy_update_kernel
  - norm_stage1_kernel
  - norm_stage2_kernel
  - normalize_kernel
  - final_projection_fanin_kernel
  - optional independent_branch_kernel to create inter-kernel parallelism

For reductions:
  - partials buffer size = partials
  - Each partial work item reduces a contiguous block:
        start = p * ceil(N / partials)
        end = min(N, start + ceil(N / partials))
    This avoids local memory and is portable.
  - stage2 can run as a single_task or parallel_for over smaller chunks.
    Prefer simple implementation first.

If using single_task causes runtime issues, use parallel_for(range<1>(1)).

================================================================================
Deliverables
================================================================================

After implementation, provide:

1. A short analysis file:
     krylov_gmres_sycl_notes.md

   Include:
     - which source repo(s) were inspected,
     - which algorithmic structure was borrowed,
     - why the final app is not a direct library port,
     - how the app maps to the three research layers:
         compute-cost semantics,
         scheduling decision,
         data-maintenance / data-visibility,
     - which modes should stress Celerity-friendly vs Celerity-unfriendly phases.

2. Source file:
     krylov_gmres_sycl_timer.cpp

3. Optional build script:
     build_krylov_gmres_sycl.sh

The build script should support something like:

  ./build_krylov_gmres_sycl.sh

and use an environment variable if available:

  CXX=${CXX:-icpx}

or

  CXX=${CXX:-acpp}

Do not hardcode one specific compiler path. Print the compile command.

Example compile commands can be included as comments:
  icpx -O3 -std=c++17 -fsycl krylov_gmres_sycl_timer.cpp -o krylov_gmres_sycl
  acpp -O3 -std=c++17 --acpp-targets=generic krylov_gmres_sycl_timer.cpp -o krylov_gmres_sycl

Use the actual local toolchain conventions only if obvious from 3mm_timer.cpp or test_copy.cpp.

================================================================================
Validation runs
================================================================================

After writing code, run at least small tests if a SYCL compiler is available:

  ./krylov_gmres_sycl --nx 256 --ny 256 --m 4 --cycles 1 --partials 128 --mode full
  ./krylov_gmres_sycl --nx 256 --ny 256 --m 4 --cycles 1 --partials 128 --mode operator-only
  ./krylov_gmres_sycl --nx 256 --ny 256 --m 4 --cycles 1 --partials 128 --mode fanin-only
  ./krylov_gmres_sycl --nx 256 --ny 256 --m 4 --cycles 1 --partials 128 --mode orthogonalization-only

If no SYCL compiler is available, at least ensure the code is syntactically coherent and explain what could not be run.

================================================================================
What not to do
================================================================================

Do not:
  - port all of Ginkgo/PETSc/Trilinos/Kokkos Kernels,
  - require MPI,
  - use USM,
  - use Celerity in this first file,
  - call wait after every kernel by default,
  - write a single-kernel benchmark,
  - implement only SpMV,
  - implement only test_copy fan-in without GMRES/Arnoldi context,
  - hide all algorithmic stages inside one giant kernel,
  - use random input without fixed seed,
  - require external datasets.

================================================================================
Expected final behavior
================================================================================

The resulting benchmark should have enough kernel structure to trigger my runtime’s wait-window DAG logic.

Specifically, in --mode full it should create a sequence like:

  stencil/operator apply
  dot reductions
  axpy updates
  norm reductions
  normalize
  repeated for k = 0..m-1 and cycles
  final projection fan-in

It should not degenerate into one large kernel or one wait per small kernel.

The benchmark should let me later write a Celerity version where:
  - operator apply can use neighborhood / one_to_one mapper and should perform well,
  - orthogonalization exposes cost imbalance / many-small-kernel behavior,
  - final projection fan-in exposes virtual-buffer coherence and data-maintenance overhead.

Proceed by first inspecting the repositories briefly, then implement the standalone benchmark.