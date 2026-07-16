# Adaptive Reacting-Flow Patch Ensemble + UQ：新正面 benchmark 设计

对应源码：

- `adaptive_reacting_flow_uq_sycl_timer.cpp`：native SYCL 与本系统共用版本；
- `adaptive_reacting_flow_uq_celerity.cpp`：Celerity 0.6.0 精确 mapper 版本；
- `build_adaptive_reacting_flow_uq_celerity.sh`：Celerity 构建入口。

## 结论

建议把这个程序作为下一轮唯一主候选，而不是继续增加彼此无关的 microbenchmark。它把
两个并行形态放进同一个物理时间步：

1. 前半段是多个独立 reacting-flow ensemble patch。每个 patch 内有规则 transport、
   空间非均匀的自适应 stiff chemistry 和 fan-in update；这里最优策略是把完整 patch
   kernel 分配给不同 GPU，而不是把每个 kernel 都按几何范围平均切给所有 GPU。
2. 后半段是一次 ensemble moments 和三阶段谱辐射风险场链。它只有一条全局链，但每个
   work-item 严格只访问自己的 cell；这里适合多 GPU Split，并让中间 partition 原地
   驻留。

这正好对应本系统的中心问题：runtime 不能只知道“kernel 可不可以切”，还要知道当前
DAG 中“应该使用 task parallelism 还是 Split parallelism”，并在二者切换时维护数据。

本 benchmark 对 Celerity 的批评不是 mapper 难写或 mapper 写错。所有 accessor 都有
精确、简单的 mapper。问题是 mapper 描述数据区域，不描述一个 chunk 中有多少 chemistry
micro-step；Celerity 0.6.0 的 coarse chunk 按几何范围分给所有本地设备，不能依据实测
成本把完整 independent task 放到设备子集。默认的 clustered flame zone 因而会让一个
或少数设备持续成为 straggler。

这仍是待实测假设，不是已经得到的性能结论。只有通过后文的 clustered/distributed/
uniform、1D/2D split、oversubscribe、单卡和消融矩阵后，才能把它升级成论文结果。

## 1. 为什么它比此前候选更合适

### Krylov / GMRES

此前 Poly 版本每个 cycle 约 92 个 kernel，存在短 fan-out，但很快重新 fan-in 到下一
basis；数万个偏小 kernel 只能放大 daemon、profile、queue rebind 和 event 查询成本。
原始 GMRES 又同时触发 reduction、host scalar、read_write、partial buffer 和长串行
basis 递推，难以把 slowdown 归到单一机制。

新程序默认每个窗口只有 `3P+4` 个 kernel：`P=8` 时为 28 个。主 chemistry 和三枚
spectral kernel 都可以通过物理参数放大到粗粒度，而不是用无效 repeat 延长。

### Reactive Transport Ensemble

此前 Reactive 的 member 内 chemistry substep 数是 member 级常量。一个 member 被按
cell 范围切分后，各 chunk 的 chemistry 工作量仍近似相等；它能说明无收益 Split 和
数据复制问题，却不能揭示“正确 access mapping 不等于正确 compute weight”。

新程序把 adaptive chemistry step count 放在逐 cell 的 `uint16` buffer 中。访问仍是
`one_to_one`，但 clustered flame zone 的 cell 要执行 1024 个 micro-step，其他 cell
只执行 8 个。成本不在 accessor shape、mode 或 mapper 中。

### SPH / composite ensemble

SPH 的 indirect neighbor gather 会迫使 Celerity 使用保守 `all` read；即使这是语义正确
的 mapper，比较仍容易被解释为数据 mapping 的特殊困难。大量 replica 和轻 EOS/
integrate kernel 也继续增加控制面比例。

新程序的主差异即使在所有 mapper 都精确 `one_to_one`/`neighborhood` 时仍然存在。它把
Layer A 的成本缺失与 Layer C 的 mapper 粗化明确分开。

### Monte Carlo radiotherapy

MC 场景已经证明 completion-driven whole-kernel scheduling 可以接近逻辑并行下界，
但每条 history 的随机早停在大量 work-item 中容易统计平均；Celerity 平分一个长 kernel
也可取得良好利用率，因此它是 Celerity-friendly control，不是相对优势案例。

新程序让高成本 cell 在空间上相关。空间相关性是 reacting-flow flame/shock/reaction
zone 的应用属性，也正是几何等分无法靠大数平均消除的条件。

## 2. 最接近的真实应用

最贴近的工作流是：

> 多个不确定性 realization 或 AMR patch realization 的 operator-split reacting-flow
> advance，随后在相同网格坐标上计算 ensemble moments，并做局部 multi-group radiation/
> risk 后处理。

它不是生产级燃烧或辐射求解器，但各阶段都有直接的应用来源：

- operator-split reactive flow 会在每个空间 cell 求一组独立 chemical-kinetics ODE；
  chemistry stiffness 随局部状态变化，GPU 实现会受到 thread divergence 和不均匀
  integration cost 影响。[Curtis、Niemeyer、Sung 的 GPU stiff-kinetics 研究](https://arxiv.org/abs/1607.03884)
  明确把 thread divergence 识别为限制因素。
- GPU chemical kinetics 通常批量求解大量 cell ODE；增加 ODE 数量和 mechanism/stiffness
  会改变适合 GPU 的执行形态。[Niemeyer 与 Sung 的 operator-split GPU kinetics 工作](https://arxiv.org/abs/1309.2710)
  给出了这种大批量独立 ODE 结构。
- patch-based AMR reacting-flow 已有 multi-GPU 框架和 chemistry integration；这支持
  “多个 patch/realization 在 coarse synchronization 前独立推进”的应用映射。
  [GPU AMR reactive-flow framework](https://arxiv.org/abs/2506.02602)。
- ensemble mean/variance 是不确定性量化的真实 fan-in；后续 spectral loop 代表局部
  nongray opacity/source quadrature。源码中的三阶段公式是稳定、可验证的 miniapp
  surrogate，不是完整 radiation transport sweep，也不包含跨 cell 的 ray/angle transport。

一个 `patch` 最稳妥的解释是一个独立 ensemble realization 的局部网格 patch，而不是
同一 AMR mesh 中必须立刻交换 halo 的相邻 patch。这样 patch 之间在 moments 前无依赖
是应用语义，不是为了调度器伪造独立性。

## 3. 完整 kernel DAG

对 patch `p` 和物理步 `t`：

```text
temperature/species[p,t]
       |                 |
       v                 v
 Transport[p,t]     Chemistry[p,t]
       |                 |
       +--------+--------+
                v
           Combine[p,t]
                |
                v
      temperature/species[p,t+1]
```

所有 patch 完成后：

```text
temperature[0..P-1,t+1]
              |
              v
       EnsembleMoments
       (mean, variance)
              |
              v
        SpectralOpacity
              |
              v
         SpectralSource
              |
              v
          SpectralRisk
              |
        semantic step wait
```

依赖类型：

- `Transport[p]` 与 `Chemistry[p]` 只读同一个 current state，彼此无依赖；
- 二者分别写私有临时 buffer，`Combine[p]` 对它们有 RAW；
- 不同 patch 使用完全不同的 buffer identity，无跨 patch RAW/WAR/WAW；
- `EnsembleMoments` 读取每个 patch 的 next temperature，对全部 `Combine[p]` 有 RAW；
- 三个 spectral kernel 是严格 RAW chain；
- ping-pong state 避免 in-place stencil 的 snapshot 歧义。

`P=8` 的 full window：

- kernel 数：`3P+4=28`；
- 最大宽度：root 层 `2P=16`；
- 关键路径：`Transport/Chemistry -> Combine -> Moments -> Opacity -> Source -> Risk`，
  共 6 层；
- 默认四个 step 共 112 个 kernel，不再是数千或数万枚小 kernel。

## 4. 为什么 wait 是算法边界

程序只在完整 macro step 的 risk field 产生后调用一次 `queue.wait_and_throw()`。下一
macro step 使用已经耦合完成的 next state；外部控制/输出也只在 risk diagnostic 完成后
观察该 step。因此这个 fence 是时间推进和 in-situ diagnostic 的语义边界。

同一窗口内没有为了 IPC 大小或调度器实现而插入 wait。`--wait-each-kernel 1` 只用于
诊断，不能作为正式性能配置。

重复窗口提供：

- 第一个窗口的 cold/derived estimate；
- 后续窗口的 Single/Split/device/mode profile；
- completion jitter 下重新派发 ready patch；
- 相同 kernel shape 的稳态 placement 和 partition scheme。

## 5. 计算几何与真实工作量

### 5.1 Transport

- NDRange：`range<2>(rows, cols)`；
- 每个 item 对 temperature、fuel、oxidizer、product 做五点 stencil，patch 边界采用
  zero-normal-gradient/clamped index；
- 计算 diffusion Laplacian 和一阶 centered advection gradient；
- 写四个同坐标 RHS field；
- work 基本均匀、memory/stencil dominated；
- 这是 Celerity 的强项，四个输入可用精确 `neighborhood(1,1)`。

### 5.2 Adaptive Chemistry

- NDRange：同上；
- 每个 item 读取自己的四个 state 和 `chemistry_steps[id]`；
- 用 reduced reversible Arrhenius kinetics 在同一个 macro chemistry interval 内执行
  `steps` 个 micro-step；
- 每个 micro-step 更新 fuel、oxidizer、product 和 temperature；
- 增加 step 数提高热区积分稳定性，所有迭代都影响最终状态，不是 busy loop。

默认工作分布：

- clustered：左上角 `rows/4 × cols/4`，即 1/16 cell 为 hot；
- hot cell：1024 micro-step；
- cold cell：8 micro-step；
- 每 patch 平均：`(1024 + 15×8)/16 = 71.5` step/cell。

`distributed` 把相同数量的 `32×32` 级 hot flamelet 均匀散到几何 chunk；`uniform`
让每 cell 为 71 或 72 step。三者总 micro-step 数相同，buffer、DAG 和 mapper 相同，
只改变 compute-cost spatial distribution。

### 5.3 Combine

- 逐 cell 组合 chemistry state 和 transport RHS；
- clamp 只用于保持 miniapp 数值稳定；
- 全部输入输出 `one_to_one`；
- 它是较轻但真实的 operator coupling phase。

### 5.4 Ensemble moments 与 spectral chain

- Moments 读取 4 或 8 个 patch temperature，同坐标计算 mean/variance；
- Opacity、Source、Risk 每个 cell 遍历 `spectral_samples`；
- spectral coefficient 是小的 read-only table，默认两张表合计仅 8 KiB；
- 增大 `spectral_samples` 表示更细的谱 quadrature，而不是无效 repeat；
- 三枚 spectral kernel 都足够规则，可作为 Celerity-friendly phase 和本系统
  SplitResident 正例。

正式运行应先在目标 GPU 上把 chemistry 和每枚 spectral kernel 校准到至少几十毫秒；
若要触发当前 runtime 的 cold Split probe，单枚 spectral kernel 还必须超过该构建的
`SNMD_OFFLINE_COLD_SPLIT_MIN_SINGLE_COST` 对应时间。默认参数只是候选，不是硬件无关
的时长保证。

## 6. 内存与访问几何

每 patch 有：

| 数据 | 数量 | 类型 | 说明 |
|---|---:|---|---|
| ping-pong state | 8 fields | `float[rows,cols]` | 4 variables × 2 versions |
| transport RHS | 4 fields | `float[rows,cols]` | stencil branch output |
| chemistry state | 4 fields | `float[rows,cols]` | adaptive branch output |
| chemistry steps | 1 field | `uint16[rows,cols]` | per-cell integration cost |

因此每 patch 是 `66 bytes/cell`。全局 statistics/radiation 是 5 个 float field，另有
两个 `spectral_samples` 长度 coefficient table。默认 `1024², P=8` 约 0.53 GiB，程序
启动时用 checked arithmetic 打印精确估算。

所有二维 field 都是 row-major；dim0 是完整 row。沿 dim0 Split 时，每个 part 写连续、
互不重叠的 row block。没有 offset、sub-buffer 或列式写。

## 7. Split 安全性与 partition contract

### Transport

Transport 读取半径 1 stencil，不能把 read accessor 声明为 partition-local。源码只对
四个 output 标记 `ext_snmd_partition_local`：写集合确实是 own block；输入仍必须用完整
replica/materializing Split，直到 runtime 有显式 halo contract。

### Chemistry、Combine、Moments、Spectral

这些 kernel 的大 field 全部严格同坐标访问，可以逐 accessor 声明 partition-local。
小 spectral coefficient table 是 replicated read，不声明 local。

预期 persistent chain 是：

```text
Moments 通常先在一台设备产生 canonical mean/variance
  -> Opacity 首次 Split，只给每卡 own mean/variance block
  -> opacity partitions 原设备驻留
  -> Source 用相同 device set 直接消费
  -> source partitions 原设备驻留
  -> Risk 用相同 scheme 消费
  -> window fence / host read 时才 materialize final risk
```

若 Opacity 不够重、profile 判断 Split 无收益，正确结果是 Single，不能为了覆盖机制强制
Split。应增加真实 `spectral_samples` 或报告该 phase 未达到 admission 门槛。

源码通过 SFINAE 调用实验扩展：在修改 runtime 中声明合同，在 stock SYCL 中自动 no-op，
因此 native baseline 使用相同算法和源码。

## 8. 本系统为什么预期表现好

### Layer A：实测成本

chemistry 的真实 service time 会进入 kernel/device/mode profile。即使 cold model 看不到
`chemistry_steps` buffer 的内容，宽 DAG guard 已阻止第一窗口在 16-wide root 层做盲目
gang Split；后续 profile 则能证明 whole-kernel throughput 更好。

注意：clustered/distributed/uniform 的 shape 和 accessor 元数据完全相同，持久化 profile
不能跨布局共用。正式实验必须分别设置 namespace，例如：

```bash
export SYCL_SNMD_PROFILE_NAMESPACE=arf-clustered-buildA
```

### Layer B：task 与 Split 的联合选择

patch phase 有 `2P` 个 READY root，四卡应运行四个完整 kernel，而不是一个 kernel 占四卡。
真实 completion 到达后，下一批 root 或对应 Combine 可以立刻填空；某个 patch/device
稍慢不会把整个静态计划锁死。

global tail 宽度降为 1 后，系统可以把足够重的 Opacity/Source/Risk 切到多卡。也就是说，
同一窗口不是固定“全 Single”或“全 distributed”，而是随 DAG phase 改变资源形态。

### Layer C：数据位置

同 patch 的 Transport/Chemistry/Combine 应因 producer affinity 保持在同一卡或只做必要
迁移。global tail 第一次按 block 汇集 patch outputs，之后 Opacity/Source 中间版本不再
每级 merge/rebroadcast。最终 risk 才形成 canonical version。

这三层缺一不可：只有 whole-patch placement 会在 global tail 留下一张卡瓶颈；只有
Split 会在 chemistry phase制造 straggler；只有 resident data 而没有全局选择会让
Split gang 抢走本可并发的 patch GPU。

## 9. Celerity 0.6.0 的准确行为与比较假设

工作区对手为 `code-celerity` commit
`8341c514069fb4e9b34eb4043851676df174a721`，tag `v0.6.0`。

源码事实：

1. `src/instruction_graph_generator.cc::split_task_execution_range()` 先把每个 splittable
   device task 的 execution range 按 `m_system.devices.size()` 生成 coarse chunks；每个
   coarse chunk 固定到对应 device。
2. 默认使用 `split_1d`；`experimental::hints::split_2d` 可切成二维几何块。
3. `oversubscribe(factor)` 只把每个 device 已有的 coarse chunk 继续细分；fine chunks
   仍被赋给同一个 `coarse_idx` device。源码注释把它定位为增加计算/通信 overlap，
   不是跨设备 work stealing。
4. `src/split.cc` 的 1D/2D split 依据 range、granularity 和几何形状；TODO 也明确尚未
   以 workload balance 作为 split heuristic。
5. range mapper 的合同只返回 chunk 所需 buffer subrange。官方文档同样把 mapper 定义
   为 execution chunk 到 data subrange 的函数：[Celerity range mapper 文档](https://celerity.github.io/docs/range-mappers/)。

因此，即使 mapper 完全正确，clustered chemistry 的 coarse chunks 仍不等成本。

### 9.1 四设备理论模型

令 hot/cold cell cost 为 `H/C`，hot square 占全 patch `1/16`。无论使用 1D 四条 row
chunk，还是 2D `2×2` chunk，左上 coarse chunk 都包含全部 hot square。其工作为：

```text
W_hot_chunk = N * (H + 3C) / 16
W_other     = N * 4C / 16
W_patch     = N * (H + 15C) / 16
```

四个 independent patch：

- 本系统 whole-patch：四卡各做一个 `W_patch`，一 wave 完成；
- Celerity per-task geometric distribution：每个 task 的 hot chunk 都落到同一 device；
  四个 task 后该 device 累计 `4*W_hot_chunk`。

忽略其他 phase，理论比值为：

```text
T_celerity / T_ours = 4 * (H + 3C) / (H + 15C)
```

默认 `H=1024,C=8` 时约 `3.66`。这不是性能承诺；warp behavior、device concurrency、
transport、global tail 和 runtime overhead 都会降低端到端差距。

### 9.2 公平 Celerity mapper

| Kernel | Read mapper | Write mapper |
|---|---|---|
| Transport | 四个 state：`neighborhood(1,1)` | 四个 RHS：`one_to_one` |
| Chemistry | state + steps：`one_to_one` | chemistry state：`one_to_one` |
| Combine | 所有输入：`one_to_one` | next state：`one_to_one` |
| Moments | 每个 patch temperature：`one_to_one` | mean/variance：`one_to_one` |
| Spectral | large field：`one_to_one`；coefficient：`all` | output：`one_to_one` |

不存在 indirect gather、错误 halo、overlapping write 或故意 `all` 大 buffer。

### 9.3 必须测试的最强 Celerity 版本

至少测试并取最快值：

- default `split_1d`；
- `hints::split_2d`；
- 两者分别配 `oversubscribe(2)` 和 `oversubscribe(4)`；
- clustered、distributed、uniform 三种 layout；
- 单 device Celerity，量化 distribution 本身是否有净收益；
- 如果被测分支新增 cost hint、device subset 或 task placement API，必须启用并重新审计，
  不能把 0.6.0 的限制泛化到未来版本。

Celerity 在 transport 和 global spectral chain 上应当很强：准确 neighborhood/
one-to-one mapping、virtual buffer partition 和异步数据流都适合这些阶段。论文应承认这点。

## 10. 参数和运行方式

通用 SYCL：

```bash
cd test-scenario
icpx -O3 -std=c++17 -fsycl \
  adaptive_reacting_flow_uq_sycl_timer.cpp \
  -o adaptive_reacting_flow_uq_sycl_timer
```

Celerity 0.6.0：

```bash
export CELERITY_INSTALL=/path/to/celerity-0.6-install
export CELERITY_DPCPP_TARGETS=nvptx64-nvidia-cuda  # 按目标后端调整
./build_adaptive_reacting_flow_uq_celerity.sh
```

小规模 smoke：

```bash
./adaptive_reacting_flow_uq_sycl_timer \
  --rows 64 --cols 64 --patches 4 --steps 1 \
  --cold-substeps 2 --hot-substeps 32 \
  --spectral-samples 8 --split-parts 4 \
  --layout clustered --mode full --verify 1
```

四 GPU 主候选：

```bash
export SYCL_SNMD_PROFILE_NAMESPACE=arf-clustered-buildA
./adaptive_reacting_flow_uq_sycl_timer \
  --rows 1024 --cols 1024 --patches 8 --steps 4 \
  --cold-substeps 8 --hot-substeps 1024 \
  --spectral-samples 1024 --split-parts 4 \
  --layout clustered --mode full --verify 1
```

同工作量 controls：

```bash
./adaptive_reacting_flow_uq_sycl_timer [same args] --layout distributed
./adaptive_reacting_flow_uq_sycl_timer [same args] --layout uniform
./adaptive_reacting_flow_uq_sycl_timer [same args] --mode patch-only
./adaptive_reacting_flow_uq_sycl_timer [same args] --mode statistics-only
```

若 chemistry 仍太短，优先增大 rows/cols 或在保持固定 macro interval 的前提下提高
hot/cold adaptive resolution；若 spectral 太短，提高真实 quadrature samples。不要只
增加 step 数来掩盖单 kernel 过细，因为 step 数不能摊薄每 kernel 控制面。

## 11. 正式实验矩阵

每组至少 5 次，报告 median、min 和 95% CI；两张/四张物理 GPU 分别先做 native
calibration。详细 trace 只用于短运行定位，正式 timing 关闭逐 kernel 输出。

| 编号 | 配置 | 回答的问题 |
|---|---|---|
| N0..N3 | native SYCL 固定到每张单 GPU | 最佳单卡与设备异构性 |
| O1 | 本系统强制最佳单 GPU/no Split | offline 控制面成本 |
| O4 | 本系统四 GPU 默认 | 主结果 |
| O4-NS | 本系统四 GPU、强制 no Split | global tail 的 Split 收益 |
| O4-M | 禁用 resident、每级 materialize | partition residency 消融 |
| O4-S | 静态一次性 HEFT、无 completion rolling | completion feedback 消融 |
| O4-C | cold-only/禁 persisted profile | profile/adaptation 消融 |
| C1 | Celerity 单 device | Celerity 控制面基线 |
| C4-1D | Celerity 四 device、split_1d | 默认几何分配 |
| C4-2D | Celerity 四 device、split_2d | 最佳 split geometry 候选 |
| C4-O2/O4 | 上述各自 oversubscribe 2/4 | 最强内置 hint |

所有 O/C 配置都跑：

- `clustered full`：主 application result；
- `distributed full`：同总 work、热点均衡 control；
- `uniform full`：几乎完全均匀 cost control；
- `clustered patch-only`：只测 cost semantics + scheduling；
- `statistics-only`：Celerity-friendly + resident Split control。

## 12. 必须收集的证据

### Layer A

- native chemistry chunk/whole-kernel event time；
- clustered/distributed/uniform 的总 micro-step 完全一致证明；
- 每个 device/mode 的 exact profile、预测来源、均值、标准差和误差；
- 1D/2D Celerity 每设备 chemistry active time。

若 clustered Celerity 各卡 active time相同，说明实际 chunk assignment 与模型不同，
不能继续沿用理论解释。

### Layer B

- 本系统每 kernel placement、parts、device set；
- patch phase Single/Split 数；
- global tail Single/SplitResident/SplitMaterialize 数；
- 每 GPU active timeline、overlap、idle gap；
- completion 后下一 dispatch 延迟；
- window critical path 和 wall time。

主机制成立的直接证据应是：patch phase 四卡同时运行四个完整 patch kernel，而不是只看
总 `run_sec`。

### Layer C

- cold/steady 分开的普通 H2D、D2D、D2H bytes/time；
- Opacity→Source、Source→Risk 边是否零 materialization；
- final risk materialization bytes；
- Celerity copy/send/receive/allocation/instruction 数；
- coefficient table replication量；
- patch output 到 moments/spectral partition 的第一次汇集量。

若本系统每个窗口仍把全部 patch state 回 host，或每枚 spectral kernel 都全量 merge，
应先修数据路径，不能把结果归到 Celerity。

## 13. 可证伪的论文门槛

建议在写主结论前设置以下 gate：

1. `VERIFY passed=1`，所有系统相同参数与容差；
2. chemistry 和 spectral kernel 粒度足以摊薄每 kernel 固定开销；
3. 本系统 clustered patch-only 的四卡 speedup 至少明显高于控制面噪声，并能从 timeline
   看到 whole-patch overlap；
4. Celerity clustered 的热点 device active time显著高于其他设备，而 distributed/
   uniform control 中该差异显著缩小；
5. 主 full 模式相对“所有 Celerity hint 中最快者”仍有稳定优势；若差距只存在于默认
   1D 而被 2D/oversubscribe 消除，就不能声称设计缺陷；
6. statistics-only 中 Celerity 应接近强基线；本系统必须通过 resident Split 避免把
   patch-phase 收益在规则 tail 中丢掉；
7. 把 cold window 与 steady windows 分开报告，不能只取 profile 已热后的最佳数字。

一个可操作但非预设结论的主门槛是：四 GPU full clustered 相对最佳 Celerity median
至少 `1.20×`，同时 distributed/uniform 差距明显收缩。达不到时，这个程序仍是机制
诊断 benchmark，但不能作为 PPOPP 主正例。

## 14. PPOPP 故事主线

最稳的题眼不是“我们比 Celerity 更快”，而是：

> **Geometry is not cost: adaptive selection between task parallelism and
> persistent partition parallelism for SYCL DAGs.**

故事链：

1. range/access contracts 能正确回答 chunk 需要什么数据，却不能回答 chunk 要算多久；
2. 当空间局部 stiffness 造成几何 chunk 不等成本，同时 DAG 又有多个 independent patch，
   把每个 task 分给所有设备会产生稳定 straggler；
3. wait-window DAG 让 runtime 看见可替代的并行形态：多个 whole-patch task 与单 kernel
   Split 争用同一组 GPU；
4. uncertainty/profile 决定代价，completion feedback 在实际完成后释放资源并重选；
5. DAG 收窄到规则 global tail 时，调度器切换为 Split；contract-proven partition
   residency 保持中间数据，避免每级 materialization；
6. Celerity-friendly transport/statistics controls 证明比较不是靠错误 mapper 或全面敌对
   workload。

三条系统贡献在一个应用中的角色：

- 不确定性感知模型：知道 chemistry whole/Split 和 spectral mode 的实测代价；
- completion-driven scheduling：动态填充 independent patch，吸收 device/patch 长尾；
- partition-resident Split：在 DAG 窄 tail 中安全地转换并行形态且不重复物化。

主张范围必须保持诚实：当前是单节点多 GPU、单应用、wait-delimited DAG；stencil 仍走
replicated read 或 Single，没有通用 halo；resident allocation 仍可能是 full virtual
allocation；Celerity 的 distributed-memory capacity、halo/virtual-buffer 能力仍是它的
优势。

## 15. 失败时如何解释

- 本系统和 native 都慢：kernel 数值实现或 GPU divergence 本身有问题，不是 scheduler；
- 本系统比最佳单卡慢：检查是否真的 whole-patch overlap、普通迁移和控制面；
- Celerity clustered 不慢：检查实际设备数、chunk geometry、是否有新版 cost-aware
  placement；理论假设被拒绝时应接受结果；
- distributed 仍和 clustered 一样慢：可能是 metadata/communication 而非 compute
  imbalance；
- statistics-only 本系统慢：检查 cold Split threshold、resident admission、每级 merge；
- 只有错误/coarse mapper 版本慢：不能作为论文主证据；
- 调高 hot steps 才胜但单 kernel 出现 watchdog/分钟级执行：参数不具代表性，应降低并
  用更大真实网格/机制复杂度校准。

这个设计的价值就在于每个解释都有对应 control，而不是只留下一个无法归因的总时间。

## 16. 当前验证状态

本工作区当前只有 Apple `/usr/bin/clang++`，没有可用的 DPC++/AdaptiveCpp、OpenCL
headers、Celerity install 或 GPU。因此没有伪造 native/多 GPU 性能数据。

已经完成的本地验证：

- 用 C++17 轻量顺序 SYCL API stub 对 stock-SYCL fallback 和
  `ext_snmd_partition_local` 可用路径分别执行 `-Wall -Wextra -Werror` 编译；
- 对 Celerity 0.6 API 形状做轻量 stub 编译，覆盖 split hint、oversubscribe、mapper、
  buffer fence 和完整 kernel 源码；
- `clustered/distributed/uniform × full/patch-only/statistics-only` 小规模运行全部
  `VERIFY passed=1`；
- SYCL 与 Celerity 版本在这些案例中的 `RESULT` 行逐字一致；
- `32×32, P=8, 3 steps, host-read-full` 在两个版本的 AddressSanitizer 和
  UndefinedBehaviorSanitizer 下均通过，结果一致；
- Celerity build script 已通过 `bash -n`。

仍必须在目标环境完成：

- 真实 SYCL device compilation；
- Celerity 0.6 + 同一 SYCL backend build；
- 单卡数值/时长校准；
- 四卡 timeline、placement、SplitResident 和 Celerity chunk trace；
- 后文完整性能/消融矩阵。

轻量 stub 只能验证 C++ 控制流、buffer 依赖构造方式、边界和数值等价，不能验证 device
compiler 限制、真实 runtime 数据移动或性能。
