# 当前系统可快速验证的性能改进

> 2026-07-15 更新：本文保留此前 Reactive/Split 消融记录。当前 runtime
> 已进一步实现 completion-driven ready queue、统一 single/Split candidate
> 和默认开启的受控 Split；现状以
> [runtime-scheduler-evolution.md](runtime-scheduler-evolution.md) 与
> [completion-driven-split-runtime.md](completion-driven-split-runtime.md) 为准。

## 结论先行

第一阶段“阻止无收益的持续 Split”已经达到目标。Reactive 的历史与最新数据为：

```text
native single GPU                         = 10.154259 s
old continuous-Split dual GPU             = 92.194035 s
forced no-Split offline dual GPU           = 11.512704 s
P1+P2+P3 default offline dual GPU          = 11.459888 s
default + detailed SPLIT_STATS             = 11.813451 s
default selected single/Split kernels      = 2400 / 0
```

默认修正相对旧结果快 `8.045×`，运行时间下降 `87.57%`，且 `VERIFY passed=1`。强制 no-Split 与默认策略只差 `0.46%`，统计日志证明默认策略本身已经选择零 Split；因此 2.48 TB 的历史一阶流量模型和旧 82 s 额外时间的归因得到强支持。

但默认双 GPU 仍比原生单 GPU 慢 `12.86%`。这已经是一个新问题：不再是 Split snapshot/merge，而是 whole-kernel placement、设备有效速度和离线控制面是否产生净收益。当前最值得做的仍不是优化 kernel clone、删除边界检查或恢复不受控的 out-of-order queue，而是把 P5 从 Split-only 统计扩展到整个 single-placement 路径，并建立 batch-level no-regression admission。

因此优先级应为：

1. 分别校准两张物理 GPU 的原生与 offline single-kernel service rate；
2. 增加 offline 强制最佳单 GPU 对照，拆出固定控制面开销；
3. 分列 raw profile、monitor penalty、普通数据移动和每卡 active time；
4. 多 GPU whole-kernel schedule 必须相对最佳 co-located batch 留出收益裕量；
5. 修正任何经计数器证明的 profile/device/queue 异常；
6. 用窄 DAG 单独验证 selective Split，不让 Reactive 宽 DAG 承担不适合它的 Split 正例；
7. 最后再考虑更大的 per-device version、halo 和显式多 stream 改造。

前四项只增加观测、测试候选或 batch admission，不改变 `cg.hpp` clone、CUDA Split copy、owned-range merge 和 pending lifecycle，因而不应重新打开此前的 CUDA 700 生命周期问题。

## 已实现的隔离开关

所有开关集中在 `code-llvm-sycl/sycl/source/detail/daemon/define.hpp`：

| 开关 | 默认 | 对应改进 | 作用 |
|---|---|---|---|
| `SNMD_OFFLINE_CANONICAL_MERGE` | 开 | P1 | handler 固定向调度选择的首设备 merge；daemon 只把该设备视为完整 Split producer source |
| `SNMD_OFFLINE_COLD_SPLIT_PROBE` | 开 | P2 | 只对预测单 kernel 至少约 5 s、且冷估计收益至少 30% 的任务允许一次 Split probe |
| `SNMD_OFFLINE_SPLIT_HYSTERESIS` | 开 | P2 | 已有 single/Split profile 后，Split 至少快 15% 才保留 |
| `SNMD_OFFLINE_WIDE_DAG_GUARD` | 开 | P3 | batch-local DAG depth 宽度足以填满 GPU 时优先 task parallelism |
| `SNMD_OFFLINE_COMPLETION_DRIVEN_QUEUE` | 开 | P6 | 单 daemon/rank 按真实 event completion 释放资源并补发 ready kernel |
| `SNMD_OFFLINE_TEST_DISABLE_SPLIT` | 关 | P0 | 仅禁用 `num_parts>1`，保留 offline HEFT 和双 GPU whole-kernel placement |
| `SNMD_OFFLINE_SPLIT_STATS` | 关 | P5 | 输出 daemon 预测决策以及 handler 实际 Split copy bytes/wait 时间 |

`SNMD_OFFLINE_TEST_DISABLE_SPLIT` 与详细统计宏在当前工作区均未打开。
正式性能构建仍应保持详细 trace 关闭，并用日志确认宽 DAG 的默认 guard
自己选择 `split_kernels=0`，而不是依赖强制禁用。

阈值也在同一文件：

```text
SNMD_OFFLINE_SPLIT_MIN_GAIN_PERCENT=15
SNMD_OFFLINE_SPLIT_THROUGHPUT_MARGIN_PERCENT=15
SNMD_OFFLINE_COLD_SPLIT_MIN_SINGLE_COST=500000
SNMD_OFFLINE_COLD_SPLIT_MIN_GAIN_PERCENT=30
```

P4 没有额外建立第二张 residency table：当前 daemon 的 `kernel_dag_nodes` 会跨 wait window 保留最近 producer 节点及其 `exec_rank/exec_proc`，candidate communication cost 已能使用这一 placement。P1 的 canonical source 修正同时适用于跨窗口 Split producer。若后续发现 DAG 生命周期或 host write 使该信息不可靠，再改为 handler 显式发送 version/device，而不是现在叠加一套可能冲突的推断状态。

统计开关打开后：

- `SNMD_SCHED_DECISION`/`SNMD_SCHED_STATS` 是 daemon 预测值，字节字段带 `estimated_`；
- `SNMD_SPLIT_STATS` 是 handler 实际走过的 direct D2D、staged D2H/H2D、merge bytes 和 host wait 时间；
- 正式性能计时应关闭该开关。

最新运行还暴露了统计层级的问题：`SNMD_OFFLINE_SPLIT_STATS` 会为 2400 个 kernel 各输出一行 `SNMD_SCHED_DECISION`，相对无统计默认版本慢 `3.09%`。后续应把“每 window 一行 summary”和“逐 kernel trace”拆成两个开关；summary 可以用于消融，trace 只用于短规模诊断。

建议按下面的宏组合做消融，每次修改后同时重编译 `sycl_object`/handler 和 `sycl-daemon`，保证两端看到相同的 canonical-merge 开关：

| 组合 | 宏配置 |
|---|---|
| 当前旧策略复现 | 关闭 `CANONICAL_MERGE`、`COLD_SPLIT_PROBE`、`SPLIT_HYSTERESIS`、`WIDE_DAG_GUARD` |
| P0 whole-kernel 对照 | 打开 `TEST_DISABLE_SPLIT`；其他 Split 修正对该运行不产生决策影响 |
| P1+P2 | 打开 canonical、cold Split probe、hysteresis；关闭 wide-DAG guard |
| P1+P2+P3 | 使用 `define.hpp` 当前默认配置 |
| 数据量定位 | 在任一组合上额外打开 `SPLIT_STATS`，但不作为正式计时 |

### 最新消融结论

- P0 已完成：强制 no-Split 为 `11.512704 s`；
- P1+P2+P3 默认组合为 `11.459888 s`，与 P0 等价地选择 2400 single、0 Split；
- 直接被 Reactive 验证的是 P3 wide-DAG admission；没有实际 Split 时，P1 merge source 与 P2 Split hysteresis 的执行路径并未被覆盖；
- 1949/451 的 kernel 数分配看似不均，但稳态是 device 2 承担一个完整 member pipeline，跨设备依赖仅约 `1.17%`，说明 affinity 基本正确；
- 最后一个 window 的合成预测成本约为 device 1 `70.80 ms`、device 2 `102.35 ms`，关键路径已经接近原生每 step 的 `101.54 ms`；
- 当前 P5 只能证明 Split bytes 为零，不能解释 device 2 为什么显著更慢，也不能统计普通 buffer migration。

## 已测事实、代码事实与推断

### 已测事实

- `8192×8192, 100 steps` 完整结束并 `VERIFY passed=1`；
- native 单 GPU 为 10.154259 s；
- 旧持续 Split 双 GPU 为 92.194035 s；
- 当前默认双 GPU 为 11.459888 s，强制 no-Split 为 11.512704 s；
- 统计运行选择 2400 single、0 Split，所有 Split copy/wait counter 为零；
- single placement 总数为 device 1 上 1949、device 2 上 451；稳态基本把 member 3 完整放在 device 2；
- 已知 DAG 边按 placement 计算的跨设备比例约为 1.17%，绝大多数集中在冷启动前几个 window；
- 初始化时间约 51.9 s，两个版本基本相同，不在 `run_sec` 内。

### 当前代码事实

- `estimateSplitInternalCopyCost` 把 dependent read bytes 从内部复制中扣除；
- 开启 canonical merge 后，`estimateCommCostForDevices` 只把 scheduler 选择的 merge device 作为 Split predecessor 完整 source；
- handler finalization 在同一个 canonical merge device 形成完整 current version；其他 part device 通常只有自己的 owned write rows；
- profile key 按 `(kernel_key, rank, device, num_parts)` 分桶；
- `num_parts=2` 的实测不会自动生成 `num_parts=1` 的 baseline；
- `estimateSplitExecCost` 在没有 Split profile 时用 `best_single/(parts×0.85)+copy`；
- HEFT 对当前节点选择 earliest finish candidate，一个 Split candidate 会同时推进所有 occupied GPU 的 available time；
- stable in-order queue 和两阶段 wait/merge 是当前正确性路径的一部分。
- single profile 优先读取 event 的 `command_start/end`，但最终 `single_exec_cost` 还会叠加 capability scaling 和 `monitorPenalty`；
- 当前 `SNMD_SPLIT_STATS` 不统计普通 non-Split buffer movement，也不分列 raw profile 与 monitor penalty。

### 需要计数器确认的推断

- 历史 2.48 TB 是按每次 Split 都重建完整 secondary snapshot 得到的一阶上界，不是硬件 counter；
- static velocity replica 可能被复用，实际 bytes 会略低；
- 历史 direct P2P 与 staged D2H→H2D 的比例仍未知；
- 当前约 1.31 s 相对原生差距中，离线 IPC/profile、普通数据移动和 window wait 的精确占比未知；
- device 2 的合成预测成本明显较高，但 raw event time、monitor penalty、硬件状态和外部占用尚未分列，所以不能断言是物理硬件变慢。

后续实现不应把推断写成已经测得的事实。

## P0：先做 split-disabled 双 GPU 对照

### 做法

临时让 `makeSplitCandidate` 返回 invalid candidate，或增加一个 daemon/runtime 环境开关：

```text
SYCL_OFFLINE_MAX_SPLIT_PARTS=1
```

保留以下所有机制：

- SCHEDULE_OFFLINE wait-window DAG；
- HEFT kernel ordering；
- 两台 GPU 的 single-device candidate；
- device rebind；
- profile 收集；
- data residency 和普通 D2D movement。

只关闭 `num_parts>1` 候选，不能通过只运行 native queue 代替，否则无法验证 task placement。

### 为什么这是最高优先级

Reactive 每步有六个独立 member，最大宽度为 12，而设备数只有 2。即使完全不 Split，也存在充分的双 GPU 工作。关闭 Split 能一次性回答：

> 当前 92 s 的主要问题究竟是 Split 数据路径，还是整个 offline scheduler 即使只做 whole-kernel placement 也很慢？

### 预期和判据

- 若时间从 92 s 大幅降到接近或低于 10.15 s，2.48 TB 假设得到强支持；
- 若低于 8 s，说明 whole-member task parallelism 已有真实收益；
- 若仍接近 92 s，应立即停止修改 Split，转而检查普通 device migration、queue wait 和 profile IPC；
- 若两台 GPU 仍不重叠，检查 HEFT placement 和跨窗口 affinity，而不是恢复 Split。

这一步是实验开关，不应作为最终“永远禁用 Split”的设计。

### 实测结果

P0 已完成：`11.512704 s`，相对旧 92.194035 s 快 8.008×，但仍比原生慢 13.38%。因此“持续 Split 是旧 slowdown 主因”已经回答；P0 之后要隔离的是 offline 最佳单卡与 offline 双卡之间的差异，不能继续把 11 秒平台归因给 Split。

## P1：修正 Split predecessor 的完整版本位置

### 原始不一致（已修正）

daemon 的 `estimateCommCostForDevices` 当前逻辑近似为：

```cpp
if (pre_node->num_parts > 1)
  source_procs = pre_node->split_devices;
```

这相当于认为 Split producer 完成后，每个 part device 都持有可供任意 consumer 完整读取的输出。

handler 的实际语义是：

1. 每个 part 只写 owned rows；
2. finalizer 把各 owned rows 合并到 merge device；
3. `MCurContext` 指向最终选出的 merge device；
4. secondary allocation 不能被视为完整 current replica。

因此对 `R(split) -> X(split)`：daemon 可能认为 GPU 0/1 都已有完整 chemistry，实际 GPU 1 只有自己先前写过的半区，X 启动前仍需从完整 merge version 构造 snapshot。

### 最小修正

首先把每个 Split 的 canonical merge device 显式化。建议调度决策使用 `split_devices.front()`/`exec_device`，handler 在 pending state 中保存该 queue，并让所有输出都合并到这个确定设备。若不希望改变当前 handler 的动态选择，则必须把实际 merge device 回传 daemon；不能让 daemon 自行猜测。

在 canonical merge device 已明确时，Split predecessor 的完整 source 应建模为：

```cpp
source_procs = {pre_node->canonical_merge_device};
```

然后：

- dependency communication model 对每个不在 canonical merge device 的 consumer target 收取 shared dependent bytes；
- `estimateSplitInternalCopyCost` 继续只负责没有 DAG producer 的 read bytes；
- 同一份 dependent bytes 只由 dependency model 计一次，避免与 internal-copy model 双算。

另一种等价方案是让 internal-copy model 对所有 read snapshot 计费，并从 dependency model 排除相同 bytes；不建议两个位置都加 `totalReadBytes`。

### 影响

这是成本模型修正，不改变实际 copy。它会让 X/Y/F 的 2-way Split candidate 看到当前 runtime 真正会发生的 secondary snapshot 成本，从源头减少错误 Split。

### 边界

未来若加入 per-device version/replica epoch，能够证明某个 secondary device 已有完整 current version，才可以重新把它加入 `source_procs`。不能只用 allocation 存在或曾参与 predecessor Split 作为证明。

## P2：single baseline 后才允许持续 Split

### 原始问题（已修正）

profile table 的 part 数隔离是正确的：single kernel time 和 Split end-to-end time不应混在一个 EWMA 中。但当前没有 exploration policy：

```text
num_parts=2 profile exists
num_parts=1 profile absent
```

这时 Split candidate 使用真实 wall time，而 single candidate 仍使用 cold heuristic。Reactive 非 root 多 buffer kernel 的 arithmetic-intensity heuristic 可能大幅高估 single time，于是已经很慢的 Split 仍会不断胜出。

### 最小策略

对每个 `kernel_key`：

1. 第一次出现时只允许 single candidate；
2. 得到至少一个 `num_parts=1` profile 后，才允许 cold Split exploration；
3. 第一次 Split sample 得到后，用 single 与 Split 的同尺度 end-to-end estimate 比较；
4. Split 必须至少改善 15%，否则进入 backoff；
5. backoff 例如 8 个 window 后才允许重新探索一次；
6. GPU monitor 或 workload shape 明显变化时可以提前解除 backoff。

建议状态：

```text
Unseen -> SingleBaseline -> SplitProbe -> PreferSingle / PreferSplit
```

### 为什么不建议随机探索

本系统有重复的语义 window，可以使用确定性的前两三个 window。随机 exploration 会让重复实验难以复现，也可能在 16 GB 级工作集上偶发一次极贵 Split。

### profile 计时的额外注意

当前 Split 的 `HostStart -> finalize timestamp` 可能包含：

- stable in-order queue 中等待更早 kernel 的时间；
- deferred merge 前提交的其他独立 kernel；
- batch-end 两阶段 finalization 的排队时间。

它适合描述该次请求的响应时间，但未必等于可以直接加到 HEFT available-time 上的独立 service cost。快速版本可以先用它做保守 admission；后续应把 pre-copy、part event、merge copy 的 operation time 与 scheduler waiting time分开，否则可能重复计算资源排队。

## P3：宽 DAG 优先 task parallelism

### 两 GPU 的吞吐条件

假设两个独立 kernel 的单卡时间都是 `T1`，一个 2-way Split kernel 时间为 `T2`：

- task parallel：两台 GPU 各跑一个，总完成时间约 `T1`；
- 连续 Split：两个 kernel 依次占两台 GPU，总完成时间约 `2T2`。

连续 Split 只有在：

```text
2T2 < T1
```

时才提高这两个 ready task 的整体吞吐，即单个 Split speedup 必须超过 2×。考虑复制和 merge，双卡通常很难超过理想 2×。因此 ready width 足够时，即使 `T2 < T1`，也不代表应该 Split。

### 最小 admission guard

在候选节点所在的 ready/parallel layer 中，若：

```text
ready_independent_tasks >= free_gpu_count
```

默认不考虑 Split，除非：

- 节点位于明显的关键尾部；
- 没有足够的其他 ready task；
- 或实测 Split speedup 大于占用设备数并留有 hysteresis margin。

不能只用 `batch_root_count`：Reactive 的 X/Y 和 F 不是 root，但其并行层宽度仍分别达到 12 和 6。应计算拓扑 level/ready set，或在 list scheduling 时维护尚未调度的 independent-ready 数量。

### 与 HEFT 的关系

这不是用一个固定规则替换 HEFT，而是把多资源 candidate 的机会成本加入 HEFT。当前 `makeSplitCandidate` 只比较当前 node 的 earliest finish；guard 让它同时考虑被占用 GPU 原本可以执行的其他 ready node。

## P4：跨 wait window 的 device residency

### 问题

Reactive 和下一代 multi-shot bench 都要求同一个 member/shot 的下一时间窗继续使用上一窗口输出。handler 通过 `MCurContext` 知道真实 current allocation；daemon 的当前 `SyclReqData` 却主要描述 buffer identity、mode、size 和 range，没有明确的 current device version。

如果每个新 window 的 root 被当作没有 producer 的普通 task，HEFT 可能看不到“state 已经在 GPU 1”，只能靠重复的调度顺序偶然保持 affinity。

### 较快的 daemon-only 方案

按 `(pid, mem_pointer)` 维护完成窗口后的完整版本位置：

```text
single writer -> exec_device
split writer  -> 实际 canonical merge device
host access   -> host/current state as appropriate
```

下一个 window 构图时，把 root read 的初始 placement 计入 candidate copy cost。该表必须在 buffer 生命周期结束、进程退出或 host write 后失效。

更严格的长期方案是由 handler 直接发送 current context/device/version；daemon-only inference 只能表达已经完成的标准 writer 路径。

### 预期收益

如果 P0 已经把 Reactive 分配为 GPU 0/1 各自负责一组 member，跨窗口 affinity 能把每个 member 的大 state movement 从“可能每步一次”降为“首次 placement 一次”。它也是下一代 multi-shot bench 获得稳态收益的必要条件。

### 最新证据与优先级

统计 placement 显示 step 5 后基本形成稳定 member affinity，全部已知 RAW 边只有约 1.17% 跨设备，且大多发生在冷启动。因此 Reactive 上暂时没有证据支持立即新增第二张 daemon residency table。P4 先降为“用普通迁移 byte counter 验证”：只有实际 non-Split D2D bytes 仍随 step 线性增长时，才升级为显式 `(buffer, version, device)` 状态。

## P5：加入最小性能计数器

在没有计数器前，不能区分 P2P、staged copy、merge 和 queue waiting。建议每个 wait window只输出一行汇总，详细 per-part trace 默认关闭。

当前 P5 只完成了 Split 子集：能证明本次所有 Split bytes/wait 为零，但不能解释 whole-kernel 路径。下面原计划中的 `ordinary_device_move_bytes`、每卡 active time 和 raw profile 分列仍未实现，是现在的最高优先级。

### handler 侧

```text
single_kernel_count
split_kernel_count_by_parts
split_input_copy_bytes
split_merge_bytes
split_prepare_host_wait_ns
split_part_wait_ns
split_merge_wait_ns
ordinary_device_move_bytes
```

### PI CUDA / copy path

```text
same_context_d2d_count/bytes
peer_d2d_count/bytes
peer_unavailable_count
staged_d2h_bytes
staged_h2d_bytes
```

### daemon 侧

```text
candidate_single_cost
candidate_split_compute_cost
candidate_split_copy_cost
selected_parts
profile_source = cold|single-ewma|split-ewma
parallel_slack_at_decision
```

### 输出原则

- 正式计时默认只输出 window summary 或最终累计；
- trace 编译开关不能改变调度语义；
- bytes 使用 64-bit；
- predicted 与 actual 分列，不能用估计值冒充实测值。

## P6：逐设备校准与 profile 分解

两张设备都报告 RTX 4090，但稳态合成成本表现出明显不对称。先做四个不改变调度核心的对照：

1. 原生 binary 只暴露物理 GPU 0；
2. 原生 binary 只暴露物理 GPU 1；
3. offline 强制所有 kernel 到 SYCL device 1；
4. offline 强制所有 kernel 到 SYCL device 2。

每组至少跑 5 次，分别报告 median/min，并同时记录每类 kernel 的 raw event duration、GPU clock、P-state、power、temperature 和 external utilization。这样可以区分：

- 原生两卡本来就不同：调度器的异构 placement 是合理的；
- 原生接近、offline 不同：检查 private context/queue、event/device 归属和普通数据移动；
- raw profile 接近、`single_exec_cost` 不同：检查 capability scaling 与 stale `monitorPenalty`；
- raw profile 自身有冷启动尖峰：把 warm-up 与 steady EWMA 分开，并做 outlier clipping。

建议诊断输出分列为：

```text
raw_profile_cost
scaled_profile_cost
monitor_penalty
final_exec_cost
profile_samples
profile_source_device
```

不要先删除 `monitorPenalty`。如果日志证明 daemon 在每个 wait 边界读到的是陈旧 100% utilization，才将其改为外部负载项、降低权重或在专用 GPU 模式禁用；设备 service time 与外部占用不能混在一个不可解释的数中。

## P7：最佳单卡回退和多 GPU batch admission

当前已有 `applyCoLocatedGpuScheduleIfBetter`，但它只在最佳 co-located 预测时间严格小于 HEFT 时间时回退，没有为 offline 控制面、首次迁移和预测误差留 margin。下一步应先扩展统计：

```text
heft_multi_gpu_finish
best_colocated_finish
selected_finish
predicted_multi_gpu_gain_percent
actual_window_wall_ns
```

然后增加独立测试开关和正式策略：

```text
SNMD_OFFLINE_TEST_FORCE_DEVICE1
SNMD_OFFLINE_TEST_FORCE_DEVICE2
SNMD_OFFLINE_MULTIGPU_HYSTERESIS
SNMD_OFFLINE_MULTIGPU_MIN_GAIN_PERCENT
```

正式策略只在：

```text
heft_multi_gpu_finish <= best_colocated_finish * (1 - margin)
```

时使用第二张 GPU。`margin` 不能凭感觉固定：先由“offline 强制最佳单卡”相对 native 的 window overhead 和预测误差分布确定，可从 10%–15% 做消融。该 guard 的目标首先是 no-regression；真正的正面 bench 应提供远大于 margin 的收益，而不是靠阈值制造速度。

## P8：统计 summary/trace 分层

当前 `SPLIT_STATS` 逐 kernel 输出使 `run_sec` 增加约 3.09%。应拆为：

- 默认关闭的 `SNMD_OFFLINE_STATS_SUMMARY`：每 window 一行计数、每卡总成本和实际 wall time；
- 默认关闭的 `SNMD_OFFLINE_STATS_TRACE`：逐 kernel candidate/rejection/profile 分解；
- 正式计时：两个都关闭，使用硬件 profiler 或只在运行结束输出累计值。

这不是主要性能优化，但它决定后续数据是否可信。特别是“统计开关不改变决策”不等于“统计开关不改变时间”。

## P9：单独验证 selective Split

Reactive 的每层 width 6/12 已足够填满两卡，默认 `split=0` 是 P3 的预期结果。P1/P2 需要另一个窄 DAG 控制：一个或少数超大 kernel、可安全 dim0 Split、single profile 已存在，并且统计能够覆盖 snapshot/merge。通过标准是：

- cold window 先 single；
- 只有 end-to-end Split 明确超过吞吐 margin 才继续 Split；
- canonical merge source、actual bytes 和后继读取位置一致；
- 强制 no-Split、selective Split 和持续 Split 三者可消融。

不要为了让 Reactive 出现 Split 而放宽 wide-DAG guard；那会重新制造刚刚消除的问题。

## 不建议作为第一步的改进

### 恢复 out-of-order 多 queue

它可能提高同 GPU overlap，但当前 HEFT 每设备只建模一条 available-time 资源。直接恢复会重新引入模型外并发、错误归因和 data race 风险。应在 compute/transfer stream 成为显式资源后再做。

### 重新启用 split-to-split replica shortcut

secondary allocation 存在不等于它含有完整 current version。没有 per-device epoch 和 in-flight transfer 状态时，这会重新打开此前的正确性漏洞。

### 只优化 cloneForSplit

CG/accessor/scalar 深复制是必要生命周期修正，host copy 很小，不可能解释 82 s。除非 profile 证明 clone 成本显著，否则不应优先。

### 删除 CUDA context/range validation

这些判断开销远小于 kernel/copy，且对定位异步 700 很重要。正式版本可以关闭日志，但没有必要删除验证语义。

### 直接实现 halo-aware Split

halo-aware snapshot、per-device version 和局部 merge 是长期重要方向，也可能让 WENO/波场 stencil 真正适合 Split。但它涉及 Layer C 数据模型重构，不属于“迅速、低风险”的第一轮性能修正。

## 下一阶段实验矩阵

所有实验使用相同 binary、相同输入和关闭详细 trace 的配置：

| 编号 | 模式 | 已知/目标 | 目的 |
|---|---|---:|---|
| A0 | native，只用物理 GPU 0 | 需测 | 每卡 raw baseline |
| A1 | native，只用物理 GPU 1 | 需测 | 判断设备差异是否真实 |
| B1 | offline，强制 SYCL device 1，`parts=1` | 需测 | 隔离 offline 固定开销 |
| B2 | offline，强制 SYCL device 2，`parts=1` | 需测 | 检查 queue/profile/device 路径 |
| C | offline 双 GPU，强制 `parts=1` | 11.512704 s | whole-kernel HEFT 对照 |
| D | P1+P2+P3 默认策略 | 11.459888 s，2400/0 | 当前正确默认 |
| E | D + summary counters | 需测 | raw/penalty/bytes/active time 归因 |
| F | E + multi-GPU batch hysteresis | 需测 | 最佳单卡 no-regression |
| G | 窄 DAG selective-Split 控制 | 需测 | 独立覆盖 P1/P2，而非改变 Reactive |

每组至少报告：

- `run_sec`；
- single/Split kernel 数；
- Split 与 ordinary input/merge/staged bytes；
- 每 GPU raw kernel active time、clock/power 和 monitor penalty；
- 两 GPU overlap；
- best-colocated、multi-GPU predicted 和 actual window time；
- cold step 0 与 steady-state step 10–99；
- checksum 和 `VERIFY`。

## 成功标准

第一阶段 admission 修正已经满足正确性、停止持续 Split 和相对 92.19 s 大幅下降三个目标。下一阶段的成功标准是：

1. Reactive 保持 `VERIFY passed=1`；
2. 默认 Reactive 继续保持 2400 single、0 Split，且 detailed stats 关闭时结果稳定；
3. 能解释两张 4090 的 raw service rate、monitor penalty 和最终成本差异；
4. ordinary movement bytes 与跨窗口 placement 一致，不存在未建模的每-step 全量迁移；
5. multi-GPU admission 相对 offline 最佳单卡不回退，预测收益和实际收益方向一致；
6. 若硬件与 workload 确实允许，双 GPU steady-state 快于 native 10.154259 s；若不允许，策略应自动选最佳单卡而不是制造 slowdown；
7. 窄 DAG 控制能解释每一次保留的 Split 为什么优于 single 与 task parallel candidate。

当前最小可发布修正仍然是 P1+P2+P3：它们已经把灾难性路径从默认执行中移除。下一版性能修正应是 P5+P6+P7，而不是先重写数据层：

- P5 告诉我们时间和 bytes 实际花在哪里；
- P6 把设备 service time 与瞬时负载分开；
- P7 保证使用更多设备必须有足够的 batch 净收益。

只有 counters 证明普通迁移或 version 缺失仍是瓶颈时才提升 P4；只有窄 DAG 证明 full snapshot 是 selective Split 的主要限制时才进入 halo/per-device version 重构。
