# Adaptive Reacting-Flow UQ v2：设计、假设与实验方法

对应文件：

- `adaptive_reacting_flow_uq_sycl_timer.cpp`：native SYCL 与本系统共用版本；
- `adaptive_reacting_flow_uq_celerity.cpp`：Celerity 0.6.0、精确 range mapper 版本；
- `build_adaptive_reacting_flow_uq_celerity.sh`：Celerity 构建入口。

本文描述 v2。v1 的高强度参数已经证明 benchmark 可以变长，但也证明它把主要工作
放在了 Celerity 最擅长的规则数据并行阶段，因此不适合作为论文主案例。

## 1. 从实测失败出发

相同四张 RTX 6000 Ada 上，v1 的极端配置为：

```text
rows=1024 cols=1024 patches=8 steps=128
cold_substeps=512 hot_substeps=65535 spectral_samples=65536
```

实测 `run_sec`：

| 模式 | 本系统 | Celerity 0.6.0 | 本系统 / Celerity |
|---|---:|---:|---:|
| full | 54.065 | 18.510 | 2.92× |
| patch-only | 17.345 | 10.932 | 1.59× |
| statistics-only | 34.046 | 8.067 | 4.22× |

Celerity 输出的 “only 2 logical cores” warning 与该结果无关；两套系统都使用了相同的
四张完整 GPU，不能再用 CPU 绑核解释差距。

`adaptive-result.txt` 还给出更直接的证据：第一窗口的 8 个 chemistry kernel 确实被
分配到 4 张 GPU，设备执行时间约为 10.3--10.8 ms，而部分 host-side dispatched
duration 达到几十至数百毫秒，与用户观察到的错峰和低利用率一致。不同 CUDA context
的 device timestamp 不应直接当作同一时钟比较。可以确定的问题不是“调度器没看到
四张卡”，而是 v1 每个 wait window 太短、窗口太多，固定控制代价和数据移动没有被
有效工作摊薄。

v1 的另一问题更严重：statistics-only 是一个规则 `one_to_one` 谱循环。Celerity 可以
把它稳定地分到四张 GPU，本系统当前却没有理由在这种短 kernel 上触发 Split。继续
只增大 `spectral_samples`，等于主动放大对手的优势。

因此 v2 不再用“更多 step + 更大 spectral loop”延长时间，而采用三项修正：

1. detailed chemistry 增加真实 reaction-channel 循环，使每个 chemistry task 粗化；
2. 减少 wait window 数，让相同量级的化学工作集中到更少的 DAG 中；
3. 加入真实的 alternating directional layout，产生精确但非 identity 的区域数据流。

## 2. 对应的真实应用

这个 miniapp 对应 patch-based adaptive reacting-flow ensemble：

- 多个 patch/sample 独立推进，代表 AMR patch、参数样本或边界条件样本；
- transport 是局部 stencil；
- chemistry 是每 cell 的 stiff source integration；
- reaction channels 代表详细或骨架化学机理中的 elementary reactions；
- ensemble moments 与 fused radiation-risk quadrature 代表 UQ 后处理；
- directional sweep 在行、列方向间切换。许多 line solve、ADI、operator-splitting 或
  tensor-product sweep 会改变存储方向，以保持当前 sweep 的访问连续。

v2 的 alternating 模式不是额外插入一个无用 transpose。transport 直接把结果写成
下一方向的存储布局；chemistry 保持当前布局；combine 在融合两条支路时完成 layout
conversion。aligned 模式计算同一离散方程但始终保持行布局，作为通信消融实验。

## 3. v2 DAG

每个物理窗口：

```text
patch p state(current layout)
      |                         |
      |                         +--> detailed chemistry (current layout) --+
      +--> directional transport (next layout) ----------------------------+--> combine (next layout)

all patch temperatures --> ensemble mean/variance --> fused radiation risk
```

对 `P` 个 patch：

- patch root：`2P`；
- combine：`P`；
- statistics：2；
- full 模式 kernel 数：`3P+2`；
- `P=8` 时每窗口 26 个 kernel、最大宽度 16、关键路径 4 层。

v1 的 opacity/source/risk 三个全局 buffer kernel 被融合成一个 kernel。三段谱数学循环
仍完整保留，结果不变，但不再让两个全局中间 buffer 和三个 task 的控制开销主导系统。

## 4. Detailed chemistry 粗化

每个 adaptive micro-step 对 `reaction_channels` 个通道计算 forward/backward Arrhenius
贡献：

```text
forward  = sum_c wf[c] * exp(-Ef[c] / T)
backward = sum_c wb[c] * exp(-Eb[c] / T)
```

权重在 host 初始化时归一化，因此改变 channel 数主要改变机理分辨率和计算量，不会让
宏观反应速率随 channel 数线性爆炸。四张 coefficient table 很小，SYCL 端不标记为
partition-local，Celerity 端使用合法的 replicated `all` read。

这一修改解决的不是 Celerity 通信问题，而是 v1 暴露出的本系统固定代价问题。论文中
应分别报告 `reaction_channels={1,32,128,512}`，展示 kernel 粒度增加后两套 runtime
如何趋近各自的数据并行或 task 并行上限。

## 5. Alternating layout 与精确 Celerity mapper

令 orientation 0 为 `A[r,c]=physical[r,c]`，orientation 1 为
`A[r,c]=physical[c,r]`。每个窗口后 orientation 翻转。

### 5.1 Transport

输出 chunk 在 next orientation，输入是 current orientation，因此读区域为输出区域转置
后向两维各扩一格的 clamped neighborhood。Celerity mapper 返回这个精确矩形：

```text
output chunk [r0:r1, c0:c1]
input region [c0-1:c1+1, r0-1:r1+1]
```

奇数 orientation 下物理 x/y 轴与存储轴相反，kernel 同时交换 up/down 与 left/right
索引；这保证 alternating 和 aligned 计算的是同一个离散方程。

### 5.2 Combine

transport 已在 next layout，使用 `one_to_one`；chemistry 仍在 current layout，使用
精确 transposed one-to-one mapper：

```text
{offset=(r,c), range=(nr,nc)} ->
{offset=(c,r), range=(nc,nr)}
```

没有任何大 buffer 使用保守 `all` mapper，也没有错误地让各设备声明重叠写入。因此若
Celerity 变慢，来源应是该应用真正要求的 redistribution、region/coherence 维护、
allocation 与 command generation，而不是不公平 accessor。

### 5.3 为什么可能区分两套系统

Celerity 对每个 task 做确定性的几何 splitting。alternating mapper 会让相邻 task 的
partition 方向交换；四设备执行时需要重新组合行/列区域，并更新 virtual-buffer 的
last-writer、replica 和 transfer 状态。8 个 patch、4 个 field、每窗口两处转置消费者，
使这种维护反复发生。

本系统可把 coarse whole-patch chemistry task 分配到不同 GPU，并让同一 patch 的后继
优先靠近其数据。对 whole task 而言，转置只是设备内索引变化，不要求把一个 patch
按行重新分发给所有 GPU。若 transport 与 chemistry 被放到不同 GPU，combine 仍会付出
一条支路的移动代价，因此优势不是无条件成立，必须由 timeline 验证 placement。

这正好体现两种抽象的区别：range mapper 精确描述“哪些数据”，但不表达 task 的空间
成本分布，也不在多个独立 patch 与单 task splitting 之间做全局 DAG 资源选择。

## 6. 默认 publication candidate

无参数默认值也是第一组候选 base：

```bash
./adaptive_reacting_flow_uq_sycl_timer \
  --rows 512 --cols 512 --patches 8 --steps 32 \
  --cold-substeps 64 --hot-substeps 8192 \
  --reaction-channels 512 --spectral-samples 1024 \
  --split-parts 4 --layout clustered \
  --directional-layout alternating --mode full \
  --wait-each-kernel 0 --host-read-full 1 --verify 1
```

该配置的静态规模：

- cells/patch：262,144，其中 hot cells 16,384；
- chemistry micro-steps/patch/window：149,946,368；
- reaction-channel eval/window：614,180,323,328；
- 32 窗口共 832 个 kernel；
- 估计 buffer 内存：145,768,448 bytes，约 139.0 MiB；
- v1 极端配置约 575 MiB，因此 v2 内存约为其四分之一。

这只是根据 v1 profile 选出的候选，不是未经测量的性能结论。若单次运行超过约 2 分钟，
优先把 `reaction_channels` 降到 128；若小于约 10 秒，先增到 1024，而不要增加 steps。

## 7. 必须做的消融矩阵

所有系统使用相同数值参数、相同四张 GPU、release build、至少 1 次 warm-up + 7 次计时，
报告 median、min 和 MAD。

v2 已给 transport/chemistry/combine 使用新的 kernel identity，避免直接命中 v1 记录。
正式实验仍应为每个参数组使用独立的 profile namespace，或设置
`SYCL_SNMD_PROFILE_PERSIST=0`；不要让不同 `reaction_channels` 共享旧观测后只测一次。

| 轴 | 取值 | 目的 |
|---|---|---|
| runtime | native 1 GPU / 本系统 4 GPU / Celerity 1、4 GPU | 基线与扩展性 |
| directional layout | aligned / alternating | 隔离转置数据流代价 |
| stiffness layout | clustered / distributed / uniform | 隔离空间成本相关性 |
| reaction channels | 1 / 32 / 128 / 512 | 固定控制开销与粗粒度拐点 |
| Celerity split | 1d / 2d | 给对手最优几何策略 |
| Celerity oversubscribe | 1 / 2 / 4 | 检查细 chunk 是否缓解不均衡 |
| phase | patch-only / statistics-only / full | 定位瓶颈 |

最关键的四组是：

```bash
# 主案例
COMMON="--rows 512 --cols 512 --patches 8 --steps 32 --cold-substeps 64 \
--hot-substeps 8192 --reaction-channels 512 --spectral-samples 1024 \
--split-parts 4 --layout clustered --mode full --wait-each-kernel 0 \
--host-read-full 1 --verify 1"

./adaptive_reacting_flow_uq_sycl_timer $COMMON \
  --directional-layout alternating
./adaptive_reacting_flow_uq_sycl_timer $COMMON \
  --directional-layout aligned

mpirun --bind-to none -n 1 ./adaptive_reacting_flow_uq_celerity $COMMON \
  --directional-layout alternating --celerity-split 1d --oversubscribe 1
mpirun --bind-to none -n 1 ./adaptive_reacting_flow_uq_celerity $COMMON \
  --directional-layout aligned --celerity-split 1d --oversubscribe 1
```

不要把 `COMMON` 字符串直接用于最终 artifact script；正式脚本应使用 shell array，避免
引用错误。上面只用于说明四条命令的公共参数。

## 8. 如何判读结果

### 8.1 支持论文故事的结果

至少同时满足：

1. native 1 GPU 与两套 4 GPU 输出通过验证，SYCL/Celerity checksum 在允许误差内一致；
2. reaction channel 增大后，本系统 patch-only 的 GPU timeline 显示 4 张卡有持续重叠；
3. clustered 相对 uniform 能明显区分 whole-patch scheduling 与 per-task geometric split；
4. alternating 相对 aligned 明显增加 Celerity 的 transfer/region/instruction 代价；
5. 本系统 alternating 相对 aligned 的增量显著较小，并且不是通过 host staging 完成；
6. Celerity 的 1d/2d/oversubscribe 中最快者仍落后，而不是只打败默认 hint。

### 8.2 说明 benchmark 仍不值得的结果

出现任一情况都应停止包装结论并继续诊断：

- 本系统 patch-only 仍比 Celerity 慢，且 timeline 显示 coarse chemistry 未并行重叠；
- 本系统 alternating 产生大量 patch 跨设备往返，抵消 task parallelism；
- Celerity 对 transposed mapper 生成零通信，说明实际 split 或 mapper 路径与假设不同；
- 优势只存在于 `celerity-split=1d, oversubscribe=1`；
- aligned 与 alternating 差距主要来自 kernel 内非合并访存，而不是 runtime transfer；
- 结果需要 `wait_each_kernel=1` 才出现。

如果本系统仍没有优势，优先修 scheduler 的 completion/transfer path，不应继续堆叠更多
敌对 access pattern。这个 benchmark 的价值在于同时暴露系统与 Celerity 的边界，而非
保证预定赢家。

## 9. 需要采集的证据

本系统：

- 每个 kernel 的 predicted/actual device time、host dispatch latency；
- HEFT placement、ready/completion 时间与四卡 overlap；
- movement bytes、source/destination、D2D 与 host staging；
- Single/Split/PersistentSplit 决策；
- 每窗口 daemon、handler、merge/finalize 时间。

Celerity：

- 每 task/chunk 的 device assignment；
- push/receive、split-receive、allocation 与 instruction 数；
- transposed transport 与 combine 的实际传输 bytes；
- 1d/2d/oversubscribe 下的四卡 active time；
- graph generation/maintenance 与 device execution 的分项。

硬件侧同时记录 100 ms 或更细粒度的 utilization、功耗和显存占用。平均利用率不能替代
timeline，因为短 kernel 的错峰执行可能产生相似平均值。

## 10. 构建与正确性

native/本系统版本：

```bash
clang++ -O3 -fsycl -std=c++17 \
  adaptive_reacting_flow_uq_sycl_timer.cpp \
  -o adaptive_reacting_flow_uq_sycl_timer
```

Celerity 0.6.0：

```bash
export CELERITY_INSTALL=/path/to/celerity-0.6-install
export CXX=/path/to/clang++
./build_adaptive_reacting_flow_uq_celerity.sh
```

短 correctness case：

```bash
./adaptive_reacting_flow_uq_sycl_timer \
  --rows 16 --cols 16 --patches 4 --steps 3 \
  --cold-substeps 1 --hot-substeps 3 --reaction-channels 2 \
  --spectral-samples 4 --split-parts 4 --layout clustered \
  --directional-layout alternating --host-read-full 1 --verify 1
```

至少检查：

- steps 为奇数和偶数；
- aligned 与 alternating；
- clustered、distributed、uniform；
- full、patch-only、statistics-only；
- patches 为 4 和 8；
- host-read-full 为 0 和 1；
- Celerity 1d/2d 与 oversubscribe。

alternating 要求 square patch，程序会拒绝 `rows != cols`。这是当前 miniapp 的明确限制，
不是 Celerity 限制；若将来支持矩形网格，需要让两个 orientation 使用互换的 buffer
shape，而不是放宽检查后继续索引。

## 11. 可主张的论文叙事范围

若实验成立，核心表述应是：

> 在具有多个独立 stiff patch、空间相关成本和交替数据布局的 wait-delimited SYCL DAG
> 中，全局 runtime 可以在 whole-task placement 与 partition parallelism 间选择，并利用
> completion feedback 和数据亲和性；仅依靠每 task 的几何 range mapping 无法同时表达
> 跨 task 资源选择与动态计算成本。

不要主张 Celerity mapper 不正确、不能表达 transpose，或 Celerity 没有使用四张 GPU。
它能精确表达这些区域，代价来自其执行与 coherence 机制在此 workload 上的选择。也要
明确 Celerity 的跨节点、halo、distributed-memory capacity 是本系统当前不具备的能力。

当前代码已用顺序 CPU stub 做过严格警告编译和小规模数值对拍：SYCL/Celerity、aligned/
alternating 在奇数步 full-read case 的 `RESULT` 一致。真实 DPC++、四 GPU 性能和 Celerity
通信 trace 仍必须在目标 Docker 环境完成，本文不把预测写成测量结果。
