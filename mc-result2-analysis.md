# `mc-result2.txt` 分析与 PPoPP 贡献成熟度

## 1. 结论

`mc-result2.txt` 中偶见的 GPU 空洞不是 monitor 造成的。四个 epoch 的所有 GPU 都满足：

```text
ServiceTimeScale=1
AvailableTime=0
```

第二个 epoch 起即使 NVML `Util=100` 或 `Util=0`，`FreshProfile=1` 都使 service scale 保持 1。daemon 每个 epoch 也都为四张 GPU 各分配两个 single-device kernel，没有任何 GPU 被 placement 阶段跳过，且 `num_parts=1` 符合宽 DAG 的设计预期。

真正异常是跨 epoch 换卡后出现的执行空洞。日志形成了很强的对照：

| Epoch | scenario 换卡数 | 实际时间线特征 |
|---|---:|---|
| 1 | 冷启动 | 每卡两个 kernel，只有约 0.16–0.90 s 普通间隙 |
| 2 | 6/8 | GPU2 第一、第二个 kernel 间约 5.74 s；GPU3 第一项也明显晚启动 |
| 3 | 4/8 | 出现 `GPU4 -> 其他三卡 -> GPU4 -> 其他三卡` 的约 5 s 波次 |
| 4 | 0/8 | 四卡重新形成连续的两波执行，异常大空洞消失 |

这说明 monitor 的 `0/100` 是前一批长尾的观测结果，不是长尾原因。

正确性结果符合预期：`VERIFY passed=1`、无 nonfinite，32 个 kernel 均完成。性能数值本身还不能与前一份 native A6000 结果直接比较：该 native 命令是 `scenarios=4, histories_per_item=256`，本日志是 `scenarios=8, histories_per_item=1024`；需要补同参数 native baseline。

## 2. 调度与实际执行对照

配置：

```text
scenarios=8
epochs=4
histories_per_item=1024
DAG width=8
GPUs=4
```

所有 32 个 kernel 都选择 single-device。每个 epoch 的静态 HEFT 计划始终是每卡两个 kernel，因此“跳卡”不是 daemon 少分配了任务。

### Epoch 2

上一 epoch 与当前 epoch 的 scenario placement 对比：

```text
scenario 7: GPU1 -> GPU1
scenario 6: GPU2 -> GPU4  migrate
scenario 5: GPU3 -> GPU2  migrate
scenario 4: GPU4 -> GPU3  migrate
scenario 3: GPU1 -> GPU1
scenario 2: GPU2 -> GPU4  migrate
scenario 1: GPU3 -> GPU2  migrate
scenario 0: GPU4 -> GPU3  migrate
```

HEFT 把约 100 MB dependent data 的纯 D2D 时间估计为约 4 ms，因此 GPU 间 profile 只要相差极小就可能换卡。但 handler 的实际数据准备受 source/destination queue 顺序约束：若目标 kernel 的 copy source queue 已经排入一个约 5 s 的无关 kernel，该 copy 以及后续 host submission 都可能被拖到该 kernel 之后。

### Epoch 3/4 对照

Epoch 3 仍有 4 次换卡，并出现约 5 s 的波次。Epoch 4 placement 完全保持，实际四卡时间线随即恢复连续。这比单独观察 NVML utilization 更能定位因果关系。

还需要下一次日志用以下新 trace 最终确认 handler 是否同步阻塞：

```text
Offline submit kernel_count=<n> host_duration_ns=<duration>
```

若换卡 kernel 的 host duration 接近一个 kernel 时间，就证明跨 context `Scheduler::addCG`/数据准备阻塞了 handler，导致尚未 resubmit 的独立 kernel 无法及时插入其他 GPU。

## 3. 本轮 runtime 修正

### 3.1 多资源通信日历

旧 candidate 只计算：

```text
target_device_ready
predecessor_finish + bandwidth_cost
```

新 candidate 为每次迁移同时建立：

```text
transfer_start = max(predecessor_finish,
                     source_device_calendar,
                     destination_device_calendar)
transfer_end   = transfer_start + communication_cost
```

并把 transfer reservation 提交回 HEFT calendar。这样通信不再被当作不占资源、随时可以发生的标量 penalty。

### 3.2 不确定性感知的迁移 admission

profile 的 Welford variance 现在真正参与 placement。每个精确 profile 的下一次运行预测不确定度由两部分构成：

```text
observed_runtime_standard_deviation
cold_start_prior / sqrt(samples)
```

第一项表示下一次执行仍会面对的运行抖动，不随样本数错误地归零；第二项表示冷启动的 epistemic uncertainty，会随精确观测增加而衰减。若迁移候选相对 data-local candidate 的预测收益没有超过联合预测区间，就保持 producer affinity。其目标不是永远禁止换卡，而是防止“为了几毫秒的点估计优势，冒险引入约 5 秒 queue disruption”；持续且显著的设备差异仍可触发重新平衡。

当前 `1.96` 是可配置的初始 operating point，不应在论文中未经实验就宣称为严格的 95% 保证；需要用 held-out epoch 检验 interval coverage，并报告不同 confidence multiplier 的性能/迁移率消融。

本轮还增加以下诊断：

```text
exec_uncertainty
movement_bytes
transfer_reservations
keeps data-local placement ... confidence_margin ... avoided_bytes
```

## 4. 下一次 radio 验证判据

使用相同参数运行，不能修改 benchmark barrier 或 scenario 数：

```text
--work-items 4194304 --scenarios 8 --epochs 4
--histories-per-item 1024
```

日志可用同目录的只读分析器复核：

```bash
python3 test-scenario/analyze_mc_runtime.py test-scenario/<new-log>.txt
```

当前日志的自动结果为 `migrations=0,6,4,0`、`split=0`，四次 monitor snapshot 的 `scales=[1.0]`、`available=[0.0]`；epoch 3 的四张卡最大内部空洞均约 5.2--5.46 s，而 epoch 4 降为约 0 s。

应满足：

1. `VERIFY passed=1` 且 checksum 与当前结果一致。
2. monitor 仍可能打印 `Util=0/100`，但 fresh device 的 `ServiceTimeScale` 必须为 1。
3. epoch 2 起大部分 scenario 保持设备；若迁移，应有显著预测收益而非几毫秒差异。
4. `movement_bytes` 与实际换卡一致；data-local candidate 应为 0。
5. 不再出现约一个 kernel 时长的 `host_duration_ns`。
6. 每卡连续执行两个 kernel，epoch 内大空洞显著下降。
7. 正式计时需关闭 daemon/handler trace，至少重复 5 次报告 median、min、max。

## 5. 三个贡献是否达到 PPoPP 程度

当前仍不能宣称三个贡献全部成熟。

> 本文件分析的 `mc-result2.txt` 生成于 completion-driven 协议落地之前；
> 下表的“当前机制”已按 2026-07-15 的代码更新，但这份旧日志不能作为新机制的
> 性能证据。

“三个贡献”是本项目用于组织论文主线的内部标准，不是 PPoPP 官方规定的数量门槛。PPoPP 2027 CFP 明确覆盖 runtime systems、tools、applications 和 practical experience；artifact evaluation 则要求 artifact 能支撑论文中的实验主张。因此这里的“达标”指每项主张都形成可解释的机制、可复现实验、强 baseline 和消融，而不是仅有三个功能标签。参见 [PPoPP 2027 CFP](https://conf.researchr.org/track/PPoPP-2027/PPoPP-2027-papers) 与 [PPoPP 2026 Artifact Evaluation](https://ppopp26.sigplan.org/track/PPoPP-2026-artifact-evaluation)。

| 贡献 | 当前机制 | 当前判定 | 达标还缺什么 |
|---|---|---|---|
| A. 独立的 cost semantics 与不确定性感知预测 | stable kernel identity、device/mode profile、EWMA、variance、monitor context、confidence-based migration | 从“数据收集”演进为实际机制，但证据不足 | 多 workload 冷启动误差、收敛曲线、置信 calibration、去掉各项的消融 |
| B. wait-window 联合调度 | 统一 single/Split selector；单 daemon/rank 的 completion-driven ready queue；真实 completion 释放 compute gang；transfer endpoint 在 dispatch wave 内排序 | 核心机制已实现，论文证据与 multi-rank 协议尚缺 | 新日志证明 idle gap/长尾改善；与 batch HEFT、greedy/Celerity 公平比较；multi-rank completion ACK 与失败恢复 |
| C. cost-aware data maintenance | producer placement、canonical Split merge、D2D cost、只读 replica cache、materialization-aware Split profile | 比旧 canonical-only 路径更完整，但仍不足以独立成为完整贡献 | 显式 partition contract、region version、partition persistence、halo/partial materialization、真实 movement counters |

因此当前更接近“两项已有实现但仍需系统实验证明的机制贡献 + 一项已开始落地的数据层方向”，仍不是三项完成的 PPoPP contribution。

## 6. 后续演进顺序

### M1：完成 contribution A 的闭环

- profile estimate 统一输出 `{mean, uncertainty, source, samples}`；
- single、Split、跨设备迁移全部使用同一个 risk-adjusted admission；
- 记录预测值与真实 service/response time，生成 calibration 数据；
- 用相同 DAG 做 cold-only、EWMA-only、monitor-only、uncertainty-aware 消融。

### M2：完成 contribution B 的 completion-driven 调度

- **已实现** `DISPATCHED/COMPLETE/PROFILED` 语义，`COMPLETE` 与 `PROFILED` 解耦；
- **已实现** 单 daemon/rank 的 window ready queue、完成反馈补位和 Split gang completion；
- **已保持** 用户 `wait()` 为最终 fence，不跨过未完成的 epoch；
- **待实现** multi-rank ACK、超时/失败恢复和系统实验。

### M3：实现 contribution C 的 region-version data plane

- 为每个 logical buffer 记录 `(version, device, valid_region, complete/partition replica)`；
- 相同 partition 的 producer/consumer 不做 batch-end 全量 merge；
- stencil 只交换 halo，host access 才 materialize 所需 region；
- 任一版本证明缺失时回退 single-device 完整版本路径。

M2/M3 都必须 feature-gated，并保留当前 batch-static、single-device fallback；不能为了论文贡献一次性替换正确性路径。
