# Batched Covariance–Subspace：由 correlation 与 Gram–Schmidt 合成的新场景

## 结论与定位

新 benchmark 是一个 **frequency-batched covariance/subspace calibration miniapp**：每个
frequency band 独立完成样本标准化、唯一相关对的三角计算、以及反向 blocked modified
Gram–Schmidt（MGS）子空间正交化。它只组合 correlation 与 Gram–Schmidt 的核心模式，
没有再引入 stencil、Monte Carlo、稀疏矩阵或人为 sleep。

对应文件：

- `covariance_subspace_common.hpp`：两套实现共享的参数、初始化、理论工作量与验证；
- `covariance_subspace_sycl_timer.cpp`：native SYCL / 本系统版本；
- `covariance_subspace_celerity.cpp`：Celerity 0.6.0 版本；
- `build_covariance_subspace_sycl.sh`；
- `build_covariance_subspace_celerity.sh`。

这个设计针对的是此前两类最稳定的 motif evidence：

1. correlation 的正确 range mapping 仍无法表达 triangular compute cost；
2. Gram–Schmidt 的 panel/update 具有逐阶段收缩的有效工作集，且小 iteration space 的
   panel 不能自然占满所有设备。

多个 band 提供真实的 logical parallelism。本系统应把完整 band-stage 当作 whole task
放到不同 GPU，并用 completion feedback 维持负载与数据亲和性；Celerity 则会把每个
高层 task 先按连续 execution range 确定性分块。新场景的目标不是让 mapper 写错，而是
验证：**即使 mapper 精确，access geometry 仍不等于 compute cost，也不回答多个独立
task 与单 task splitting 应如何竞争同一组 GPU。**

这仍然是一项待测研究假设，不是预先写好的性能结论。若本系统不能把 band DAG 并行起来，
或 Celerity 的 concurrent scheduler 能消除长尾，结果应如实作为负结果报告。

## 1. 贴近的真实应用

最直接的应用映射是多频带/多时间块阵列信号的 RFI（radio-frequency interference）
检测与子空间校准：

- 每个 frequency channel 或 time sub-integration 形成一组多 antenna/beam samples；
- 对 samples 做中心化与方差标准化；
- 计算 beam/antenna 间的 cross-correlation / sample covariance；
- 从 correlation matrix 提取并正交化 interference subspace，供 projection、flagging 或
  后续 eigen/subspace iteration 使用；
- 不同 frequency channel/time block 彼此独立，天然构成批量 DAG。

这不是凭名字拼接。mRAID 工作明确以 multi-beam cross-correlation matrix 的
eigendecomposition 识别 RFI，并指出每个 time sub-integration 与 frequency channel 的
CCM/EVD 相互独立、可完全并行；array telescope 的 subspace projection 也以 sample
covariance 估计干扰子空间。参见 [mRAID](https://arxiv.org/abs/2512.16278) 与
[RFI subspace projection](https://arxiv.org/abs/1809.03620)。更一般地，subspace/
orthogonal iteration 在 covariance/PCA 中反复执行矩阵作用并用 QR 重新正交化中间
subspace；参见 [memory-efficient PCA 中的 subspace iteration](https://pmc.ncbi.nlm.nih.gov/articles/PMC4735350/)
和 [randomized subspace iteration](https://arxiv.org/abs/1804.02614)。

miniapp 与生产算法的边界也必须明确：

- 它存储每个唯一 correlation pair 一次，形成带对角 regularization 的上三角 row
  sketches，然后直接对这些 rows 做 blocked MGS；
- 它不是 mRAID 的完整 EVD，不声称数值输出等价于某个 telescope pipeline；
- blocked MGS 是 QR/orthonormalization 的调度替身，保留 norm、normalize、projection、
  update 与 shrinking active set；
- 使用 real FP32，而实际 radio covariance 通常是 complex；complex 会扩大常数但不会
  改变这里要测试的 DAG/range-cost 关系。

因此正确表述是“application-derived miniapp / scheduling proxy”，不是“完整 RFI
application”。若论文需要 end-to-end science result，应再替换输入与最终 detection，
但不应改掉本场景的核心 task geometry。

## 2. 数学与访问模式

对 band `b`，输入为 `X_b[a,s]`：`A` 个 antenna/features，`S` 个 snapshots。

标准化：

```text
mean[a]   = sum_s X[a,s] / S
std[a]    = sqrt(sum_s (X[a,s] - mean[a])^2 / S)
X[a,s]    = (X[a,s] - mean[a]) / std[a]
```

唯一 correlation pairs：

```text
C[w,j] = sum_s X[row(w),s] * X[j,s] / S,  j >= row(w)
C[w,j] = 0,                                j <  row(w)
```

主模式 `covariance-order=forward` 使用 `row(w)=w`。work-item `w` 的主计算量为：

```text
W(w) = S * (A - w)
```

所以执行范围虽然是均匀的 `range(A)`，计算量却是三角形。输出按 work row 连续写入，
没有 overlapping writer。输入的精确需求是 sample rows `[w,A)`；一个 execution chunk
的精确 rectangular union 是从该 chunk 最小 `w` 开始的 suffix。

随后对前 `K` rows 做反向 blocked MGS。对 panel `[p,e)`：

1. 一个 work-group 在 panel 内顺序选择 pivot，并用 128 lanes 做 norm/dot reduction；
2. update kernel 把所有 `r < p` 的 rows 对该 orthonormal panel 投影并更新；
3. 下一个 panel 向低 row index 推进，有效 prefix 逐阶段缩短。

主模式 `compact-gs=0` 保持 update execution range 为 `range(K)`，只让 `r<p` 工作。这与
原始 Gram–Schmidt 的 `range(N)` + `if(j<=k) return` 模式一致。Celerity mapper仍只声明
真实 prefix writes；因此空 work-item 不制造数据移动，只制造无法从 range 看出的计算
不均衡。`compact-gs=1` 提交 `range(p)`，数学和访问结果不变，是把 cost knowledge 显式
编码进 iteration geometry 的控制组。

## 3. DAG 与同步窗口

一个 band、一个 epoch 的 DAG 为：

```text
Mean -> Stddev -> Normalize -> TriangularCorrelation
                                      |
                                      v
Panel(last) -> Update(prefix) -> ... -> Panel(first)
```

默认 `A=768, S=2048, K=512, panel=64` 有 8 个 panels：

- correlation 部分 4 个 kernels；
- 8 个 panel kernels；
- 最低 panel 没有 prefix，所以 7 个 update kernels；
- 每 band 共 19 个 kernels；
- 8 bands 时每 wait window 152 个 kernels，最大逻辑宽度 8；
- 3 epochs 提供 3 个语义一致的 wait windows，让系统从 cold estimate 进入 live profile。

SYCL 版本只在 epoch 末 `queue.wait_and_throw()`；Celerity 版本只在 epoch 末
`slow_full_sync()`。`wait-each-kernel=1` 是故意摧毁 DAG width 的负控制，不是正式配置。

每个 band 使用独立 buffers，依赖不是靠 kernel 名字或提交顺序猜测。branch-major
submission 只是 host 顺序；正确 runtime 应从不相交 memory objects 看到 band independence。

## 4. 为什么这个场景可能区分两套系统

### 4.1 Correlation 的理论负载上限

当 `A` 可被 4 整除、Celerity 以连续 1-D chunks 均分 `w` 时，四块的三角工作占比趋近：

```text
43.75%, 31.25%, 18.75%, 6.25%
```

单 task 的 4-GPU speedup ceiling 因而约为：

```text
1 / 0.4375 = 2.286x
```

8 个相同 band 即使可以跨 task 并发，所有 task 的最重 contiguous chunk 仍映射到同一
coarse device，瓶颈累计不消失。本系统若把 whole bands 以两波分到四张 GPU，理想吞吐
是 4x，单从三角负载就有约 `4 / 2.286 = 1.75x` 的相对空间。

程序会打印实际整数规模的 `THEORY covariance_chunk_shares`，因此非整除规模不需要使用
渐近值。

### 4.2 Gram–Schmidt 的 shrinking prefix

masked GS update 每阶段仍提交 `range(K)`，但有效 prefix 从 `K-panel` 逐步缩到
`panel`。连续 coarse chunks 的累计工作再次呈三角形；低-index device 长时间承担工作，
高-index device 逐步空闲。

panel kernel 只有一个 work-group，但内部 reduction 使用 128 lanes，并非单 GPU thread
串行循环。一个 panel task 本身不值得跨设备；关键是同一 DAG depth 有 8 个独立 band
panels。本系统应一张 GPU 放一个 whole panel task，Celerity 的 range-driven local split
无法把“多个 range 只有一个 work-group的独立 task”自动改写成 whole-task device
placement 搜索。

### 4.3 精确 mapper 仍有真实数据维护

Celerity 版本没有对 device kernel 使用虚假的 `all` mapper：

- mean/stddev：chunk antenna rows × all snapshots；
- normalize：data one-to-one，mean/stddev 是对应 antenna slice；
- triangular correlation input：chunk 内所有 `row(w)` suffix 的精确 rectangular union；
- correlation output：chunk work rows × all columns，互不重叠；
- panel：精确 fixed panel rows；
- update target：chunk 与 active prefix 的精确交集；
- update pivot：精确 fixed panel rows。

每个 update 的 panel rows 必须对所有 active chunks 可见，这是算法真实的 pivot broadcast/
replica requirement。Celerity 对 last-writer、allocation、replica 和 instruction graph 的
维护可能有额外成本，但论文必须把它与 triangular compute imbalance 分开归因。

### 4.4 Oversubscription 为什么不是默认解法

当前 `code-celerity/src/instruction_graph_generator.cc` 先按 device 数产生连续 coarse
chunks，再把每张 device 自己的 coarse chunk 细分为 oversubscribed fine chunks；fine
chunks 不会重新分配到其他 device。因此 `oversubscribe` 可以增加 copy/compute overlap，
但不改变 43.75/31.25/18.75/6.25 的 coarse ownership。

正式实验仍必须报告 `oversubscribe={1,2,4,8}` 并取 Celerity 最快值，不能仅凭源码推断
性能。将来 Celerity 若加入动态 chunk stealing，这一结论也必须重测。

## 5. 本系统的预期决策与边界

主配置中 `bands >= GPU count`，每一 DAG depth 都有足够 whole tasks：

- wide-DAG guard 应拒绝无 profile 的 gang Split；
- unified selector 应把同一 stage 的不同 bands 分到空闲 GPU；
- completion-driven dispatch 应在真实较快 GPU 空闲后继续派发，而不是固化静态 queue；
- producer locality 应尽量让同一 band 的后继留在拥有 `data/basis` 的 GPU；
- epochs 1/2 应使用 live profile，但 stage scalar `panel_begin` 不在 profile key 中，所以
  update profile 是不同 active-prefix cost 的混合。这是故意保留的 cost-semantics stress，
  不能靠给每个 stage 人工换 kernel 名字规避。

本 benchmark **没有**调用 `ext_snmd_partition_local`：GS update 除了写自己的 prefix row，
还读取 fixed pivot panel，不能谎称所有 reads 都是 partition-local。若 daemon 选择 Split，
只能走 materializing Split；正确的主决策应是在宽 DAG 中使用 whole-task parallelism。
因此它主要验证贡献一和贡献二，以及“拒绝不合适 Split”的能力，不是 resident Split 的
正面样例。resident Split 必须保留独立 pointwise-chain microbenchmark。

## 6. 控制模式

### `--mode`

- `full`：标准化 + correlation + blocked MGS，主案例；
- `correlation`：只保留 correlation pipeline，隔离三角负载；
- `orthogonalize`：从 deterministic upper-triangular basis 开始，仅跑一次 MGS；该模式
  要求 `--epochs 1`。

### `--covariance-order`

- `forward`：主模式，连续 work chunks 对应连续 antenna0，三角负载集中；
- `folded`：work rows 交替映射低/高 antenna0，让每个连续 chunk 混合重/轻 rows。输出
  仍按 work row 连续且数学 pair 集合不变，但输入 suffix union 更大；它是 compute
  balance 控制，不是等通信量控制。

### `--compact-gs`

- `0`：固定 `range(K)` + masked prefix，访问 mapper 仍精确；
- `1`：每阶段提交真实 `range(p)`，把 active work 暴露给 range splitter。

### `--wait-each-kernel`

- `0`：正式 wait-window DAG；
- `1`：破坏跨 band/stage DAG，可证明优势不是单个 kernel codegen 差异。

## 7. 构建与短正确性测试

SYCL / 本系统：

```bash
cd test-scenario
CXX=/path/to/clang++ \
SYCL_EXTRA_FLAGS='-fsycl-targets=nvptx64-nvidia-cuda' \
./build_covariance_subspace_sycl.sh
```

若 toolchain 默认 target 已是 CUDA，可不设置 `SYCL_EXTRA_FLAGS`。

Celerity 0.6.0：

```bash
cd test-scenario
CELERITY_INSTALL=/path/to/celerity-install \
CXX=/path/to/clang++ \
CELERITY_DPCPP_TARGETS=nvptx64-nvidia-cuda \
./build_covariance_subspace_celerity.sh
```

短 correctness case：

```bash
COMMON=(--antennas 64 --snapshots 128 --bands 4 --epochs 2 \
        --rank 32 --panel 8 --mode full --verify 1)

./covariance_subspace_sycl_timer "${COMMON[@]}"
mpirun --bind-to none -n 1 ./covariance_subspace_celerity "${COMMON[@]}"
```

两套实现应同时输出 `VERIFY passed=1`，`nonfinite=0`，checksum 在 FP32 允许误差内一致。
验证从每个 band 读取最终 basis，检查全部元素有限、加权 checksum，以及最多 12 个均匀
抽样 rows 的 norm 和 pairwise dot。host materialization 单独记为 `host_sec`，不计入
`run_sec`。

## 8. 第一组性能候选

默认参数就是第一组候选：

```bash
COMMON=(--antennas 768 --snapshots 2048 --bands 8 --epochs 3 \
        --rank 512 --panel 64 --mode full \
        --covariance-order forward --compact-gs 0 \
        --wait-each-kernel 0 --verify 1)

SYCL_SNMD_PROFILE_PERSIST=0 \
  ./covariance_subspace_sycl_timer "${COMMON[@]}"

mpirun --bind-to none -n 1 ./covariance_subspace_celerity \
  "${COMMON[@]}" --oversubscribe 1
```

它不是未经测量即可进入论文的固定规模。调参顺序应是：

1. 若单次不足 10 秒，先增 `snapshots` 到 4096；
2. 若超过 2 分钟，先减 `snapshots`，不要减 bands 到 GPU 数以下；
3. 保持 `rank/panel` 至少 6 个 panels，使 shrinking-prefix 可见；
4. kernel 仍过小时增 `antennas`/`snapshots`，不要靠增加 epochs 或拆更多 kernels 掩盖
   控制开销。

## 9. 必须做的实验矩阵

所有正式结果至少 1 次 warm-up + 7 次计时，报告 median、min、MAD；固定 GPU clocks/
power mode，关闭 daemon/handler trace。

| 轴 | 取值 | 目的 |
|---|---|---|
| runtime | native SYCL 1 GPU / 本系统 1、4 GPU / Celerity 1、4 GPU | 基线与扩展性 |
| mode | correlation / orthogonalize / full | 分离两种 motif |
| covariance order | forward / folded | 三角 compute imbalance 控制 |
| GS range | masked / compact | access 精确但 cost 隐藏 vs cost 显式 |
| bands | 1 / 2 / 4 / 8 / 16 | whole-task width 与 gang Split 竞争 |
| snapshots | 512 / 2048 / 8192 | 固定控制代价拐点 |
| panel | 16 / 32 / 64 / 128 | kernel 数与 panel/update 比例 |
| Celerity oversubscribe | 1 / 2 / 4 / 8 | 给对手最佳 local overlap |
| epochs | 1 / 3 / 8 | cold/live profile 收敛 |
| 本系统 Split | `SYCL_SNMD_ENABLE_SPLIT=0/1` | 证明 wide-DAG admission 而非 hard-disable |
| completion queue | 0 / 1 | 静态计划与真实 completion feedback |

最关键的四个对照：

```text
A. forward + masked       主案例：两种 cost blindness 同时存在
B. folded  + masked       correlation 更均衡，GS 仍隐藏 active prefix
C. forward + compact      correlation 仍三角，GS cost 显式
D. folded  + compact      对两种 runtime 都更友好的控制
```

如果差距在 D 仍同样大，不能把结论归因于 range-cost mismatch，应检查固定 runtime 开销、
queue/context、copy、profile journal 或实现 bug。

## 10. 论文通过门槛

只有同时满足以下条件，才能把它作为 PPOPP 主案例：

1. native、本系统、Celerity 输出全部通过验证；
2. 本系统 4 GPU 相对自身 1 GPU 至少 3.0x，且 timeline 显示 4 卡真实 overlap；
3. Celerity `forward+masked` 的 active time 接近理论 coarse-chunk 不均衡，而不是设备发现
   或错误 mapper 导致单卡；
4. Celerity 最佳 oversubscribe 仍未消除 coarse-device 长尾；
5. folded 和 compact 两个控制分别缩小对应 phase 的差距；
6. 本系统启用 Split 时主要选择 whole Single tasks，关闭 Split 不应突然才变快；
7. epochs 后续窗口相对 cold window 不变差，completion queue 的收益能在 ready/
   completion timeline 中解释；
8. 本系统优势不是来自 Celerity host materialization，因为 `run_sec` 与 `host_sec` 已分开。

建议把“本系统 4 GPU 比 Celerity 最佳 4 GPU 至少 1.3x”设为实用门槛；理论上三角 phase
可能有约 1.75x 空间，但 regular phases、panel broadcast 与 runtime overhead 会降低端到端
比值。不要把 1.75x 写成测量预期保证。

## 11. 失败结果如何解释

出现以下任一情况，不应继续包装主张：

- 本系统 4 GPU 不到 3x，且 handler log 显示 independent bands 被 read/read migration
  或 host `resubmit()` 串行化：先修 runtime data path；
- daemon 对宽 DAG 大量选择 2/4-way Split：检查 width 计算、profile namespace 和 unified
  risk，不要修改应用隐藏该决策；
- Celerity folded/compact 仍与主模式同速：采集 instruction/device timeline，确认 mapper、
  hint 和实际 chunks；
- Celerity 主模式接近 4x：说明 concurrent scheduling 或未来动态 balancing 已经克服该
  workload，本文比较点需要更新；
- 两套实现 checksum 不一致：先缩小到 64×128 correctness case，检查 panel accessor
  mapping 和 buffer last-writer，不能放宽 tolerance 掩盖错误；
- 优势只在 `wait-each-kernel=1` 出现：场景无效，因为正式系统故事依赖完整 DAG；
- panel kernel 占绝大多数时间：减 panel size或增 snapshots，并确认 128-lane reduction
  正常运行，不能让一个低并行 kernel 伪装成 scheduler 优势。

## 12. 必采证据

本系统：

- 每个 node 的 kernel identity、stage、band、predicted/actual device、profile source；
- READY/DISPATCHED/COMPLETE 时间和每波 movement bytes；
- Single/Split/persistent decision，尤其 wide-DAG reject；
- 每 GPU active/idle timeline、host resubmit latency；
- epoch 0/1/2 的 CostEstimate mean/uncertainty 变化。

Celerity：

- 每个 task 的 coarse/fine chunks 与 device assignment；
- correlation 四个 coarse chunks 的 kernel active time；
- masked GS 每 panel 的有效 prefix 和各卡 idle time；
- panel fixed-read replica/copy bytes、allocation/resize/copy/free/instruction 数；
- oversubscribe 1/2/4/8 的最快配置。

硬件同时记录 GPU utilization、功耗、显存和 clocks。平均 utilization 只能辅助，不能替代
per-kernel timeline。

## 13. 可主张的故事范围

若实验通过，最稳妥的主张是：

> 在具有多个独立 frequency-band DAG、三角 correlation cost 和收缩
> orthogonalization active set 的单节点多 GPU workload 中，精确 access-range mapping
> 仍不足以决定计算分区。一个把 compute profile、DAG width、data locality、Split mode
> 与实际 completion 统一起来的 runtime，可以选择 whole-task parallelism，避免连续
> range splitting 的系统性长尾。

不要声称：

- Celerity mapper 不正确或无法表达这些 regions；
- Celerity 一般不能并发 task；
- 本 miniapp 是完整 mRAID/EVD；
- 当前系统已具备 Celerity 的跨节点 virtual buffer、halo 与超设备内存能力；
- 该结果证明 resident Split，因为主案例有意不声明 partition-local contract。
