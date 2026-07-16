# mc-result3：功能贡献判断与下一步实现

> 2026-07-16 更新：本文的 `mc-result3` 实验判断不变；其后 runtime 已实现
> 下文提出的显式 partition-local contract 和 `SplitResident`。最新数据面设计与
> 验证入口见 [completion-driven-split-runtime.md](completion-driven-split-runtime.md)。

## 结论

只看功能、不把尚未完成的实验当作实现缺陷，`mc-result3.txt` 证明第二项贡献——completion-driven DAG scheduling——在单 daemon/rank、单节点多 GPU 范围内已经闭环。当前最需要补齐的不是再改 ready queue，而是第一项贡献中的统一预测/决策语义：此前 cold、profile、monitor、migration 和 Split 虽然汇入同一个 selector，仍使用不同的 uncertainty fallback、15% Split hysteresis 和 migration-only confidence rule。

本轮已将这些路径改为显式 `CostEstimate` 和统一 UCB 风险目标。随后实现的
`SplitResident` 使用明确 partition contract 和 window-local 版本状态，不在没有访问
证明时通过猜测 accessor 布局跳过 materialization。

## 1. mc-result3 的功能证据

对日志运行 `python3 analyze_mc_runtime.py mc-result3.txt` 得到：

- 8 scenarios × 4 epochs = 32 个 kernel；32 次 completion dispatch、32 次 completion done、32 个 profile，缺失、重复和未知 completion 均为 0。
- 每个 epoch 四张 GPU 都各执行 2 个 whole-scenario kernel；总 migration=0、Split=0。
- epoch 2–4 每张卡两个相邻 kernel 的最大内部空洞约 1.8–7.1 ms。设备一完成第一波工作，daemon 就派发第二波，不再等待整批第一波都完成。
- monitor 在后续 epoch 虽看到四卡 100% utilization，但 `FreshProfile=1` 使 service scale 保持 1.0、合成 available time 保持 0；自身刚完成的工作没有被重复惩罚。
- `run_sec=42.299867`，验证 `VERIFY passed=1`。单 kernel 大约 4.8–5.3 s；每个 epoch 两波、共四个 epoch，运行时间已经接近该 DAG 的约 40–42 s 逻辑下界。

epoch 2–4 的首个 profile start 在设备间仍有约 0.72–1.60 s 偏斜。这不是 completion queue 再次形成 batch barrier：同卡两波之间只有毫秒级空洞。它更可能来自 handler 顺序 `resubmit()` 时跨 context 数据准备/提交的 host-side skew；需要开启 `PRINT_HANDLER_TRACE` 后才能把它与真实 queue start 分开。该问题属于后续 submission/data-plane 优化，不否定 ready queue 的功能闭环。

## 2. 三项贡献的功能成熟度

### 贡献一：不确定性感知的冷启动与在线预测

本轮前是“组件存在但语义未统一”：已有稳定 kernel identity、EWMA/Welford variance、sample count、capability scaling、monitor freshness 和 locality admission，但只有迁移使用联合置信边界，profiled Split 使用固定 15%，cold Split 又使用 5 s/30% 门槛。

本轮实现后：

```text
CostEstimate = {mean, uncertainty, samples, source}
source = exact-profile | scaled-profile | cold | derived-split

risk(candidate) = mean_EFT
                + beta * hypot(exec_uncertainty, transfer_uncertainty)
```

- exact profile 在一次加锁读取中同时取得 EWMA、预测标准差和样本数；
- scaled profile 保留源样本抖动并增加 target capability model error，不能冒充 exact；
- cold 和 derived Split 显式携带模型不确定性；
- monitor service scale 同时作用于 mean 与 uncertainty；
- dependent transfer 产生均值和不确定性；
- HEFT upward rank、Single/Split placement、co-located fallback 和 completion 后重选都使用同一风险目标。

冷 Split 的 5 s/30% 规则仍保留，但其角色是限制未知模式 exploration 的可行域，不再是 profiled candidate 的长期 penalty。宽 DAG guard 同理是资源形态约束：已有足够 task parallelism 时，只有经 profile 证明的 throughput win 才允许 gang Split。

随后补齐的分层学习把第一项从“统一接口”推进到可跨运行积累的机制：rank 0 daemon
加载/异步追加 profile observation journal，持久化 exact 与 live exact 分源并带年龄
误差；跨设备缩放使用采样时 capability；同 kernel identity 可以跨 shape 迁移，严格
structural cohort 可以在更宽不确定度下提供最后一级 learned prior。完整回退顺序为：

```text
live exact -> persisted exact -> exact-key scaled
           -> same-identity learned -> structural learned -> cold
```

因此第一项在 runtime 功能上已形成“观测—持久化—迁移—风险决策”的闭环，但不能
仅凭功能称为“冷启动精度已经证明”。access range 已替代 backing-buffer size 进入
work/precision 特征，跨 shape 尺度代理也与当前 DAG 宽度解耦；尚缺 kernel 指令混合、
occupancy/cache/寄存器特征和 estimate-source calibration 实验。不同应用/构建应通过
`SYCL_SNMD_PROFILE_NAMESPACE` 隔离；默认 namespace 无法自动识别同名 kernel 的
二进制实现已经改变。

### 贡献二：profile-feedback completion-driven HEFT

在当前声明范围内已功能完成：

- handler completion 与 profile 分离，缺 profile 不阻塞资源释放；
- daemon 维护 PENDING/READY/DISPATCHED/COMPLETE 状态；
- single device 和 Split gang 都在真实 completion 后释放；
- handler/daemon 有 batch dispatch、completion 和 failure protocol；
- 静态 HEFT 与 rolling dispatch 共用 candidate selector；
- 用户 wait 仍是最终 fence，没有推测尚未提交的下一 epoch；
- `mc-result3` 未出现 GPU 被 completion queue 跳过、重复释放或丢 completion。

边界是 multi-rank：尚无 remote completion/data-ready ACK、超时和失败恢复，因此 multi-rank 仍走静态 fallback。这是贡献范围限制，不是单节点实现未闭环。

### 贡献三：materialization-aware Split 到 partition-resident dataflow

`mc-result3` 生成时只完成第一阶段：Split 默认开启、layout/memory guard、gang
reservation、精确 part profile、canonical merge、read-only replica cache 和失败
回退。其 DAG width=8，在四卡上不 Split 是正确的全局决策，不能用 Split 数量判断
功能缺失。

当前代码又补齐了严格 partition-local 范围内的 writable partition 跨 kernel 驻留、
per-buffer superseded version、兼容边零 materialization、首次输入按 part 搬运、
不兼容边/fence canonical fallback，以及 resident/materializing profile mode 隔离。
所以 pointwise/逐行/独立 history chain 的 `SplitResident` 已功能闭环；halo-only
transfer、任意 region read-set 和 partial materialization 仍未完成。若把贡献表述为
通用 stencil mapper，第三项仍不能称为完成；若表述为“contract-proven persistent
Split with safe materialization fallback”，机制已经成立，后续缺实验与 halo 扩展。

## 3. 本轮代码实现

daemon 新增统一估计和风险比较：

1. `CostEstimateSource/CostEstimate` 明确记录来源、均值、不确定性和样本数。
2. `lookupExactProfileEstimate()` 原子读取 exact profile 的全部统计量；`lookupScaledProfileEstimate()` 显式增加跨设备模型误差。
3. Single 和 Split estimator 都返回 `CostEstimate`；derived Split 传播 single uncertainty 并加入 Split model error。
4. dependency transfer plan 记录预测 cost/uncertainty；候选总 uncertainty 合并 execution 与 transfer。
5. `preferTaskCandidate()` 唯一比较 `mean + beta*uncertainty`；风险等价才以移动字节、设备数、均值完成时间和设备编号确定性打破平局。
6. 删除 migration-only confidence override 和 profiled Split 15% hysteresis。
7. HEFT upward rank 和 co-located batch fallback 也改用风险目标，避免后处理 heuristic 绕过统一 selector。
8. trace 增加 `cost_source`、`profile_samples`、`transfer_uncertainty` 和 `risk_score`。

## 4. 后续实现：contract-proven SplitResident

统一风险目标之后已加入显式 accessor contract，并以 handler pending state 实现
window-local buffer version ledger：

```text
ResidentBufferVersion {
  producer_kernel,
  ordered_split_devices,
  part_events,
  written_buffers,
  superseded_written_buffers
}
```

已完成的安全顺序是：显式 `ext_snmd_partition_local(accessor)`；相同 parts/device
scheme 的 Split producer→consumer 直接复用；旧 buffer version supersede；不兼容
consumer 和用户 fence 回退 canonical materialization。daemon 同时验证 access
range/offset/sub-buffer/atomic，handler 再做一次真实 requirement 检查，所以 stale 或
过度乐观的决策只能降级，不能破坏正确性。

下一层才是 halo region、interior/halo overlap 和 single/host consumer 的 partial
materialization。它们需要新的 mapper/region contract，不应继续扩张当前
partition-local 声明的语义。
