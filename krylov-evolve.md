从 GMRES 到 Poly 的演化
最早的 krylov_gmres_sycl_timer.cpp 试图构造一个 GMRES/Arnoldi 风格的真实 Krylov miniapp。它包含典型 Krylov 求解器特征：
v_k -> A v_k
A v_k -> dot/reduction -> h_ik
h_ik -> orthogonalization/update
norm/reduction -> v_(k+1)
basis -> projection/solution update
restart
这类结构很真实，但它天然带来三个问题：
Arnoldi/GMRES 每一步依赖前一步生成的新 basis vector。
dot/norm/reduction 会产生标量依赖，常常需要 host 或全局同步参与。
orthogonalization/update 很容易使用 read_write、partial buffer、scalar buffer、reused temporary buffer。
因此原版 GMRES 对当前系统来说是一个“压力测试”，不是一个好的正面案例。它同时触发了 reduction、标量依赖、read_write、split merge、跨设备数据一致性、kernel clone、mqueue 大消息等多个问题。任何一个问题出错都会导致卡住或 CUDA 700，很难判断系统调度思想本身是否有效。
为了让 bench 能在当前 runtime 边界内运行，我们把它改成了 krylov_poly_sycl_timer.cpp。
poly 版保留了 Krylov 的外观：
Q_(k+1) = mix(A Q_k, Q_0..Q_k, b, branch outputs)
x       = sum_i c_i Q_i
它仍然有：
2D stencil，模拟 matrix-free operator A Q_k
多个 basis plane Q_0..Q_m
restart cycle
basis fan-in
每个 cycle 一次 queue.wait()，让 daemon 可以基于上一轮 profile 调度下一轮
但它去掉了原 GMRES 中最麻烦的部分：
无 reduction
无 scalar Hessenberg buffer
无 host scalar decision
无 atomics
默认无 read_write
所有大数据都用严格 2D buffer
写入使用 row-block 形式，方便当前 split 逻辑处理
也就是说，poly 版已经不是生产 GMRES，而是一个“带 Krylov 命名和 basis 结构的 polynomial stencil DAG”。
当前 Poly 版的 DAG 形状
mode=full 下，每个 cycle 是：
for k = 0..m-1:
  Q_k -> stencil -> work

  Q_k, b -> branch_0
  Q_k, b -> branch_1
  ...
  Q_k, b -> branch_B

  Q_0..Q_k, work, b, branches -> candidate
  candidate -> Q_(k+1)

Q_0..Q_fanout -> projection -> x

if not last cycle:
  x -> stencil -> work
  b, work -> residual
  residual, b -> Q_0

queue.wait()
每个 stage 有：
1 stencil
B branch kernels
1 basis_mix
1 publish
= B + 3 kernels
你这次运行参数是：
nx=4096 ny=4096 m=8 cycles=300 branches=8 fanout=8
所以每个 basis cycle：
m * (B + 3) = 8 * 11 = 88 kernels
非最后 cycle：
88 + projection 1 + residual chain 3 = 92 kernels
最后 cycle：
88 + projection 1 = 89 kernels
总 kernel 数约为：
299 * 92 + 89 + final sample 1 = 27598 kernels
这就是现在最大的问题：kernel 数量非常多，但每个 kernel 都很小。
为什么它不适合作为正面案例
这个 poly 版虽然比原 GMRES 更适合当前 runtime 正确执行，但仍不适合展示系统优势。
第一，顺序依赖太强。
每个 stage 虽然有 B 个 branch 可以并行，但它们马上被 basis_mix 汇合；basis_mix 之后还要 publish 成 Q_(k+1)，下一步 k+1 才能开始。也就是说 DAG 形状是反复出现的短 fan-out / 立即 fan-in：
Q_k
 | \
 |  branch_0..branch_B
 |
stencil
 \       /
  basis_mix
     |
  publish
     |
  Q_(k+1)
逻辑并行宽度存在，但持续时间太短，不能形成足够大的调度空间。
第二，kernel 太小且偏 memory-bound。
很多 kernel 只是：
stencil：少量邻域读写
branch：简单逐元素公式
publish：纯 copy
residual：逐元素减法
projection/mix：有小循环，但仍主要读多个 buffer 后写一个 buffer
这些 kernel 对单 GPU 来说已经很快。系统每个 kernel 都做 profile、调度记录、event 查询、queue/rebind、daemon 通信，开销就会和 kernel 本身同量级。
第三，profile 开销无法摊薄。
你的结果：
单 GPU run_sec 约 8.3s
系统 run_sec 约 14.0s
额外开销约 5.7s
而总 kernel 数约 27598，则额外开销平均约：
5.7s / 27598 ≈ 0.207 ms/kernel
对于大 kernel 这点开销可以接受；但对于几万枚小 kernel，这就是致命开销。
第四，daemon 最终调度到同一 GPU 其实是合理的。
因为 DAG 串行依赖多，数据复用集中在 basis/work/candidate/residual 上。跨 GPU 执行会引入 D2D/H2D/H2D 或 split merge 成本。profile 观察到这些成本后，把大部分 kernel 调回同一 GPU，是符合代价模型的。
问题在于：系统虽然做了大量 profile 和调度工作，但最终没有可利用的跨 GPU 并行收益，于是只剩开销。
结论
原 GMRES 版太真实，触发太多 runtime 正确性边界，不适合作为正面案例。
poly 版为了适配 runtime，去掉了 reduction、标量依赖和复杂 read_write，但也因此不再是真正的 GMRES。它仍保留 Krylov 的串行 basis 递推结构，导致：
DAG 逻辑并行度不足
kernel 粒度太小
profile/daemon 开销过高
最优调度自然退化到单 GPU
系统运行时间反而接近单 GPU 的 1.7 倍
所以 Krylov 更适合作为“系统边界与负面案例”：
它说明当前 runtime 面对细粒度、强依赖、频繁 profile 的 Krylov 型 workload 时，
调度收益不足以覆盖系统开销。
不适合作为正面展示。
正面案例应该重新构造，目标是：
每个 kernel 足够粗，至少毫秒级
每个 cycle 的 kernel 数不要太多
DAG 有持续的逻辑并行，而不是立即 fan-in
多个 GPU 有明确可利用的独立任务
profile 决策可以跨 cycle 复用并摊销
数据移动/merge 成本低于并行收益
更合适的方向是多分支独立任务型 miniapp，例如多场景 stencil、ensemble simulation、多 block PDE、batched pipeline，或者 3mm 那种结构清晰、kernel 粗、DAG 分支明确的计算图。