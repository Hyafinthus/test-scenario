# 长 kernel 正例：多情景 Monte Carlo 光子输运

对应源码：`mc_radiotherapy_scenarios_sycl_timer.cpp`。

## 1. 结论与设计边界

这个 benchmark 专门隔离当前系统最可能获得净收益的一种情况：一个 wait window 里只有少量、彼此独立、单次持续数十秒的 kernel。每个逻辑分支拥有完全独立的 source、material 和 tally buffer；双 GPU 的主要机会是把不同完整 kernel 放到不同设备并发执行，而不是把毫秒级 kernel 的固定开销继续放大。

程序不是用空循环、固定 `repeat` 或无效算术把 kernel 拖长。每个 work-item 完整模拟多条 photon history；每次碰撞都执行自由程采样、分层材料查找、连续吸收和 Henyey–Greenstein 散射方向更新。增加 `histories_per_item` 对应增加真实 Monte Carlo 样本数，统计误差原则上随样本数下降，而不是只增加不改变问题的 busy work。

这个程序仍然只是调度 miniapp，不是经过临床验证的剂量引擎。它没有电子次级粒子、真实截面表、三维 CT voxel、全局 dose-grid 原子累加和最终情景汇总；不能把其数值结果解释成临床剂量。

## 2. 为什么贴近真实应用

最贴近的真实工作流是放疗计划的 Monte Carlo 鲁棒性评估或 4-D 剂量计算：

- GPU Monte Carlo 剂量计算已有“一个 GPU thread 模拟一条完整 photon history”的实现先例；大量 history 是获得统计精度所必需的。[Hissoiny 等的 GPUMCD](https://pubmed.ncbi.nlm.nih.gov/21452713/)和[早期 GPU 放疗 Monte Carlo 实验](https://pmc.ncbi.nlm.nih.gov/articles/PMC2884304/)都属于这条路线。
- 鲁棒性评估会对病人摆位、解剖变化和设备机械误差形成的多个 uncertainty scenario 分别做 Monte Carlo dose calculation；这些 scenario 在最终比较/汇总前是独立的大任务。[Monte Carlo robustness tool](https://pmc.ncbi.nlm.nih.gov/articles/PMC9545707/)明确描述了多个用户定义不确定性情景的自动计算。
- 4-D 放疗剂量要在不同呼吸相位的 CT 上分别计算剂量，随后再配准和累加。[4-D Monte Carlo dosimetry](https://pmc.ncbi.nlm.nih.gov/articles/PMC2890615/)给出了这种“各呼吸相位先独立计算、最后融合”的结构。
- miniapp 使用的 Henyey–Greenstein 散射是组织光子输运 Monte Carlo 中常见的相函数；这里借用的是输运计算形态，不声称替代高能放疗的完整相互作用模型。[凝聚历史 photon transport 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC2424239/)讨论了组织输运中的 HG 相函数和 photon biography。

因此程序中的一个 `scenario` 可以解释为一个 setup uncertainty、一个呼吸相位或一个束流/材料参数 realization。每个情景使用独立数组是应用语义，而不是为了给调度器伪造依赖。

## 3. Kernel DAG

每个 epoch 只提交 `S` 个 `PhotonTransport[s]`：

```text
PhotonTransport[0]    source/material/tally[0]
PhotonTransport[1]    source/material/tally[1]
...
PhotonTransport[S-1]  source/material/tally[S-1]
             \          |          /
              epoch semantic wait
```

同一 epoch 内：

- RAW：无跨情景 RAW；每个 kernel 只读自己的 source/material/tally。
- WAR/WAW：无跨情景 WAR/WAW；buffer identity 全部不同。
- 独立分支：`S` 条。
- 最大宽度：`S`。
- 关键路径：1 个 kernel level。
- kernel 数：每个 window 为 `S`，整个运行是 `S * epochs`。

跨 epoch 时，同一情景的 tally 使用 `read_write` 累积，因此有真实的上一批 tally 到下一批 tally 的 RAW/WAW 关系。不同情景始终没有依赖。

默认 `scenarios=4, epochs=2` 时总共只有 8 个 timed kernel；这与 Krylov Poly 的约 2.76 万个 kernel、Reactive 的 2400 个 kernel 和 SPH 的 3.84 万个 kernel 是有意的数量级差异。

## 4. 同步边界

一个 epoch 表示一批独立 Monte Carlo histories。程序在同一 epoch 的所有情景提交完成后调用一次 `queue.wait()`，原因是：

1. 形成一个完整统计 batch；
2. 让上一窗口的 device/profile 结果可用于下一窗口；
3. 为同一情景的后续 tally 累积建立明确完成边界。

情景 kernel 之间没有 wait。`--wait-each-kernel 1` 只是诊断串行路径，不能作为性能结果。

## 5. 计算几何与长 kernel 标定

每个 kernel 使用一维 `range(work_items)`。work-item `i` 负责：

```text
for history = 0 .. histories_per_item-1
  初始化 photon packet 和 RNG
  for collision = 0 .. max_collisions-1
    按当前 z 定位材料层
    用 -log(U) / mu_t 采样自由程
    推进三维位置
    连续吸收一部分权重
    用 Henyey-Greenstein 相函数采样新方向
    出界、权重耗尽或达到 collision cap 时结束
  累积 thread-private tally
写回 item i 的 tally
```

计算是不规则的：history 可能因反射、透射或低权重提前终止；相同 `range` 大小不意味着每个情景或每个设备上完全相同的时间。程序打印每 kernel 的 history 数和最坏 collision-step 上界，但这两个数不能替代实测 kernel 时间。

“至少数十秒”必须在目标 GPU 上标定，不能仅凭源码常量宣称。程序提供只运行一次然后退出的标定模式：

```bash
./mc_radiotherapy_scenarios_sycl_timer \
  --calibrate 1 \
  --work-items 262144 \
  --histories-per-item 8 \
  --max-collisions 128 \
  --target-kernel-sec 20
```

正式运行的候选默认值是 `histories_per_item=256`，有意设置得很重；标定命令显式从 8 开始，避免第一次探测就运行数分钟。默认值仍不是硬件无关的 20 s 保证，最终门槛只认目标 GPU 的实测。

输出中的 `recommended_histories_per_item` 按一次实测线性外推。正式实验要把这个值固定下来，并在 native 两张物理 GPU 上分别用 event profile 复测；只有较快 GPU 的 kernel 第 25 百分位也达到 20 s，才能声称满足设计门槛。不要让 native、offline 和 Celerity 各自重新标定成不同工作量。

当前 daemon 的 profile key 不含 `histories_per_item` 这类 scalar capture。为避免一次短标定污染随后长 kernel 的 profile，标定模式内部使用 `layers+1` 的材料表，使 accessor shape 与正式 kernel 不同；输出的正式命令仍保留用户给定的 `layers`。最稳妥的流程仍是在 native 构建上标定，然后对所有系统使用固定参数。

推荐两步法：

1. 用 native 最快单卡运行 `--calibrate 1`；
2. 将推荐值写入所有被比较系统的同一条正式命令。

长时间 kernel 在启用图形 watchdog/TDR 的设备上可能被驱动终止；正式运行应使用没有短 watchdog 的 compute GPU。

## 6. 内存几何

每个情景有 12 个独立的一维 buffer：

| Buffer | 元素数 | Mode | 含义 |
| --- | ---: | --- | --- |
| `seeds` | `N` | read | 每个 work-item 的 RNG seed |
| `source_energy` | `N` | read | photon 初始权重 |
| `source_cosine` | `N` | read | 入射方向余弦 |
| `absorption` | `L` | read | 分层吸收系数 |
| `scattering` | `L` | read | 分层散射系数 |
| `anisotropy` | `L` | read | HG 参数 `g` |
| `deposited` | `N` | read_write | 累积吸收权重 |
| `reflected` | `N` | read_write | 累积反射权重 |
| `transmitted` | `N` | read_write | 累积透射权重 |
| `residual` | `N` | read_write | collision cap 后剩余权重 |
| `path_length` | `N` | read_write | 累积路径长度 |
| `collision_count` | `N` | read_write | 累积碰撞数 |

所有 `N` 长度数组的 accessor 都覆盖完整 `[0,N)`；材料表覆盖 `[0,L)`。每个情景的预计内存是 `36*N + 12*L` bytes。默认内存很小是刻意的：这个实验希望测长计算 kernel 的 task placement，而不是显存容量或大规模传输。

## 7. Split 安全性与预期决策

沿 NDRange dim0 做 `P` 份 Split 时：

- 必须满足 `N % P == 0`，程序通过 `--split-parts` 提前检查；
- work-item `i` 只读写 tally 元素 `i`；
- 每个 part 写连续且互不重叠的 `[p*N/P,(p+1)*N/P)`；
- row-major 一维布局与当前 handler 的 dim0 owned-range merge 完全一致；
- 小材料表是全读，分发到每个参与设备是安全的。

所以这个 kernel 是“可安全 Split”，不是通过 guard buffer 人为禁止 Split。但主模式已经有 `S>=设备数` 个长时间 ready kernel，合理决策应优先 whole-kernel placement：两卡各执行不同情景，不为 read_write tally 制造 snapshot 和 merge。`scenarios=1` 是专门验证 selective Split 的窄 DAG 控制，不应与多情景主结果混报。

## 8. 对当前系统的直接预期

结合 `daemon.cpp` 和 `handler.cpp` 当前实现，默认第一窗口有以下有利条件：

- 四个节点都是 batch root，HEFT 的 device available time 应把它们分散到两张 GPU；
- 每个情景的 buffer identity 不同，不需要在情景之间搬数据；
- bounded cold Split probe 只允许预测 single 至少约 5 s、且模型收益至少
  30% 的窄 DAG；本例四情景宽 DAG 仍应由 wide-DAG guard 保持 whole-kernel；
- `SNMD_OFFLINE_WIDE_DAG_GUARD` 应在宽度已经填满 GPU 时继续偏向 whole-kernel throughput；
- 第二个 epoch 的 profile key、NDRange 和 accessor 形状不变，可以复用第一窗口实测；相同 shape 的四个情景会共享/汇总 profile，而不是按 buffer 内容分别建模；
- tally 的跨窗口依赖应让同一情景保持 producer affinity，避免把完整 read_write 数组迁到另一设备。

如果每个 kernel 为 20 s，四情景单 epoch 的理想时间下界近似为：

```text
最佳单 GPU：4 * 20 s = 80 s
双 GPU whole-kernel：ceil(4 / 2) * 20 s = 40 s
```

这只是理想模型，不是已测结果。若系统仍显著慢于单 GPU，毫秒级固定开销已经不能解释，应该直接检查：两设备是否真实重叠、是否把所有 kernel 放到同一卡、是否在 root kernel 前发生整 buffer host round-trip、private device queue 是否串行等待另一设备，以及长 kernel 是否被错误 Split。

## 9. 数据移动预期

第一次 epoch：

- 每个情景的 source、material 和零初始化 tally 从 host materialize 到其被选设备；
- 情景之间无 D2D；
- 若保持 single placement，无 Split merge。

稳态 epoch：

- source/material 是只读且可驻留；
- tally 的 current version 应留在同一情景的设备上；
- 合理 affinity 下，正常计算前的 H2D/D2D 应接近零；
- final host accessor 的 materialization 计入单列 `host_sec`，不计入 `run_sec`。

正式结果至少记录每设备 assigned kernel 数、kernel active time、overlap、普通 H2D/D2D/D2H bytes、Split 数和 window wall time。仅凭总 `run_sec` 无法证明发生了 kernel-level 并发。

## 10. Celerity 映射与公平解释

这个案例本身对 Celerity 很友好：

- execution range 是 `[0,N)`；
- source 和所有 tally 都可用 `one_to_one` mapper；
- `absorption/scattering/anisotropy` 是很小的 fixed/all-read 材料表；
- 写集合与 execution chunk 完全同构；
- 不需要 halo，也没有跨情景 coherence。

Celerity 可以把单个情景按 work-item range 分布到所有 worker；额外成本只有小材料表复制、初始/最终 tally 可见性和 task/command 管理。由于单 kernel 非常长，这些成本容易被摊薄。因此本 benchmark 是验证“当前系统能否终于靠粗粒度 task parallelism 超过 native 单 GPU”的正例，也是 Celerity-friendly 控制；它不适合单独支持本系统优于 Celerity的论文结论。

公平 Celerity 实现若支持 worker subset/task hint，也应测试：一种策略是每个情景固定一个 worker，另一种是把每个情景分到全部 worker。取较快者作为对手。

## 11. 三层实验假设

### Layer A：计算成本

假设：单 kernel 的实测时间主要由真实 photon histories 和碰撞数决定，20 s 以上时控制面开销可忽略；当前 shape-based profile 学到的是这一组相似情景在各 device/part 上的平均 service rate。若要研究材料内容导致的强异构成本，需要让情景具有不同 NDRange/buffer shape，或扩展 profile key/应用标签，不能声称当前 key 能区分相同 shape 的 buffer 内容。

证据：native 每 kernel event 时间、碰撞数、情景/device/part EWMA、预测误差。若推荐参数仍只有毫秒或数秒，先增加 `histories_per_item`，不能继续做系统对比。

### Layer B：调度决策

假设：宽 DAG 上的最佳策略是 whole-scenario placement；双 GPU window wall time应接近每卡 assigned kernel 时间总和的最大值，而不是所有 kernel 时间之和。

证据：每 kernel placement、`num_parts`、两卡 overlap 和 window makespan。主模式如果持续 Split 或所有 kernel 在同一卡，假设被拒绝。

### Layer C：数据维护

假设：独立 buffer 加跨窗口 affinity 后，初始化之外的普通迁移和 merge 很少，数据维护不会抵消数十秒计算重叠。

证据：cold/steady 分开的 H2D、D2D、D2H、merge bytes/time，以及最终 host materialization。若每 epoch 重复搬完整 tally，应先修 residency/version 路径。

## 12. 编译、运行与实验矩阵

通用 DPC++ 示例：

```bash
cd test-scenario
icpx -O3 -std=c++17 -fsycl \
  mc_radiotherapy_scenarios_sycl_timer.cpp \
  -o mc_radiotherapy_scenarios_sycl_timer
```

小规模 smoke test：

```bash
./mc_radiotherapy_scenarios_sycl_timer \
  --work-items 1024 --scenarios 2 --epochs 2 \
  --histories-per-item 2 --max-collisions 16 \
  --layers 16 --host-read-full 1 --verify 1
```

标定后正式主模式示例；`H` 必须替换为标定输出：

```bash
./mc_radiotherapy_scenarios_sycl_timer \
  --work-items 262144 --scenarios 4 --epochs 3 \
  --histories-per-item H --max-collisions 128 \
  --layers 128 --verify 1
```

实验矩阵：

| 编号 | 配置 | 回答的问题 |
| --- | --- | --- |
| A0 | native，物理 GPU 0，4 scenarios | 最佳单卡基线候选 |
| A1 | native，物理 GPU 1，4 scenarios | 两卡 service rate 校准 |
| B0 | offline 强制 GPU 0，no Split | 长 kernel 下固定控制面成本 |
| B1 | offline 强制 GPU 1，no Split | 第二设备/queue 路径校准 |
| C | offline 双 GPU，强制 no Split | 纯 whole-kernel placement |
| D | offline 双 GPU，默认 selective Split | 宽 DAG guard 是否选择正确策略 |
| E | `scenarios=1`，native/offline Split | 窄 DAG 的 intra-kernel 控制 |
| F | Celerity，准确 mapper 和最佳 hints | Celerity-friendly 对照 |

每组至少 5 次，关闭逐 kernel 详细 trace，报告 median/min。主模式成立的最低门槛是：`VERIFY passed=1`、每个物理 GPU 上单 kernel P25 大于等于 20 s、双卡存在真实 overlap、稳态无反复整 buffer 迁移、默认策略没有无收益 Split，并且双卡 `run_sec` 显著低于最佳 native 单卡。未完成这些测量前，只能称为“按目标构造并通过代码验证的 benchmark”，不能写成已有加速结果。
