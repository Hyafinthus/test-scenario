# 当前贡献缺口、路线选择与实现状态

## 结论

本轮暂不使用实验结果判断贡献，只检查 runtime 是否形成了可定义、可组合、不会被
其他 heuristic 绕过的机制。

三个贡献目前适合按下面的范围表述：

| 贡献 | 已形成的功能闭环 | 仍不能声称的范围 |
|---|---|---|
| 不确定性感知预测 | profile 统计、持久化、分层迁移、来源/年龄不确定度和统一风险决策 | compiler-level 静态性能模型、冷启动精度已经由实验校准 |
| 负反馈 DAG 调度 | 单 daemon/rank 内 completion 驱动释放与重选，profile 与 completion 解耦 | multi-rank rolling dispatch、远端失败恢复 |
| 持久化数据 Split | 显式 partition-local contract 下，分区版本可跨相邻 kernel 驻留，不兼容边安全物化 | 自动推断 stencil halo、跨 wait/跨 rank 永久分布式 buffer、按分区物理分配显存 |

因此，如果论文声明范围是“单节点多 GPU + contract-proven partition dataflow”，三条
都已有对应的 runtime 功能核心；如果声明为“通用多节点/通用 stencil runtime”，第二、
第三条仍不完整。实验只能在这个功能边界确定后验证收益，不能用一次性能提升替代缺失
的协议或访问证明。

## 1. 本轮为何先补分层预测

此前统一 selector 已经能接收 `{mean, uncertainty, samples, source}`，但 daemon 重启
后仍从 analytical cold 开始，也不能使用同一 kernel 的其他 shape。这样“更精确的
冷启动”仍只是接口设计，不是可积累的系统机制。

本轮选择先补这一层，原因是它同时满足三个约束：

1. 直接补齐第一项贡献中最明确的功能缺口；
2. 不改变 handler/daemon 调度协议和数据正确性路径；
3. profile 存储完全脱离 completion 临界路径，不重新引入此前已删除的等待。

没有选择直接加入 multi-rank rolling dispatch，是因为它必须同时定义远端 completion、
data-ready、幂等和失败状态；只增加一条 ACK 会产生比静态 fallback 更危险的半完成协议。
也没有让 daemon 自动推断 halo，因为 accessor/NDRange 元数据不能证明 kernel 内索引关系。

## 2. 已实现的自适应预测层

### 2.1 统一命中顺序

Single 和 Split estimator 现在按相同顺序查找：

```text
live exact
  -> persisted exact
  -> same exact key on another device, capability-scaled
  -> same kernel identity at a nearby problem shape
  -> strict structural cohort at a nearby shape
  -> analytical cold model
```

`num_parts` 与 `persistent_split` 始终是 mode 边界。Single、materializing Split 和
resident Split 的 observation 不会混桶。

同 identity 的迁移用于相同 kernel 的 shape scaling；不同 identity 只有在 work
dimension、requirement 数量、读写数量、access mode 集合、主元素宽度和 partition
contract 完全一致时才能成为 structural neighbor。若存在 identity neighbor，预测器
不会再混入 structural neighbor。

### 2.2 不确定度不是标签

每种来源都把不确定度送入现有统一 UCB 目标：

- live exact：Welford 观测波动与随样本数衰减的 prior；
- persisted exact：在上述项外增加 25% stale floor，并按每天 1% 老化；
- device scaling：传播源 profile 波动，再增加 capability model error；
- same-identity learned：传播 profile 波动，并增加 20% 与 shape 距离项；
- structural learned：使用 60% 起步的模型误差，只能作为 cold 前的弱先验；
- 多 neighbor：二阶矩同时保留单 neighbor uncertainty 与 neighbor disagreement。

因而旧样本可以帮助冷启动，但不能以“历史上运行过”为由绕过当前 monitor memory
feasibility；只有本进程的 live exact 才能作为曾在当前设备配置成功分配的证据。

### 2.3 可迁移的 shape 特征

持久化 feature 不复用 `coldWorkElems()`。后者为宽 DAG root 刻意改变 arithmetic
intensity guard，若直接写入 profile，会让相同 kernel 因当前窗口宽窄不同而得到不同
身份特征。

现在使用：

```text
shape_work = max(global_items, total_access_elements)
```

它只负责同 kernel 的问题规模缩放；kernel 自身算术强度来自实测 service time。跨
kernel 的算术强度无法从这些字段可靠获得，所以 structural transfer 保留很宽的风险
区间。下一阶段若需要更强的跨 kernel cold model，应由编译器提供 instruction mix、
register/shared-memory、occupancy 与静态访存信息，而不是继续增加 daemon 猜测规则。

### 2.4 非阻塞持久化

rank 0 daemon 启动时读取 observation journal；kernel 完成后只在已有 profile lock 内
更新内存统计，释放锁后把 observation 放入有界队列。独立 writer thread 负责 append
和 flush：

```text
kernel COMPLETE
  -> update in-memory EWMA/variance
  -> release profile lock
  -> bounded enqueue
  -> scheduling may immediately admit READY work

background writer
  -> append observation journal
```

队列满时丢弃最旧的待落盘 observation，而不是阻塞 completion；当前进程的内存模型
仍已更新。退出时才 drain 并 join writer。记录保存采样时间、采样设备 FP32/FP64
capability、exact profile key、part/mode 和静态 feature；跨重启设备缩放使用记录能力，
不把本次进程中相同 rank/device 编号误当成原设备。

运行时配置：

```bash
# 建议使用应用版本、git commit 或二进制哈希
export SYCL_SNMD_PROFILE_NAMESPACE=myapp-build-id

# 可选；默认写 /tmp 下的 rank0 journal
export SYCL_SNMD_PROFILE_STORE=/path/to/profile-observations.tsv

# 正式 cold-only 消融
export SYCL_SNMD_PROFILE_PERSIST=0
```

namespace 会写入每条 V2 record。旧 V1 record 只属于 `default`，自定义 namespace 不会
加载它。默认 namespace 保留兼容性，但无法检测 kernel 名称不变而实现已修改，因此不
适合跨版本正式实验。

## 3. 还缺的贡献设计

### 3.1 Multi-rank completion-driven queue

这是第二项贡献剩余的最大功能缺口。正确路线不是把现有 completion message 简单经
MPI 转发，而是由 master 持有唯一 node state，并加入 dispatch lease：

```text
DISPATCH(window_id, node_id, lease_id, target_rank, devices, input_versions)
EXEC_ACCEPTED(window_id, node_id, lease_id)
COMPUTE_COMPLETE(window_id, node_id, lease_id, output_versions)
DATA_READY(window_id, node_id, lease_id, regions)
```

设计约束：

1. `(window_id, node_id, lease_id)` 幂等，重复/迟到 ACK 不能释放新 reservation；
2. `COMPUTE_COMPLETE` 只释放 compute gang，跨 rank consumer 必须等 `DATA_READY`；
3. master 的 ready queue、device lease 和 buffer-version ledger 是同一状态事务；
4. 任一动态 dispatch 发出后不得静默切回静态路径；协议失败应使用户 wait 失败；
5. timeout 不能盲目重跑有副作用的 kernel。只有输入版本未变且输出尚未 publish 时才
   可换 lease 重试，否则 fail-stop；
6. 启用动态协议前进行 capability negotiation，旧 daemon/rank 整个 window 使用静态
   fallback。

建议实现顺序是：单远端 rank 无数据边测试 → compute/data-ready 分离 → 多远端 rank
→ timeout/late ACK → daemon failure。每一步都保留整个 window 开始前的静态 fallback，
而不是在状态机中途降级。

### 3.2 SplitHalo 与 region version

当前 `ext_snmd_partition_local()` 的语义是严格 own-block，不允许 stencil 假报。下一层
应扩展为显式 mapper descriptor，而不是复用一个布尔值：

```text
PartitionAccess {
  kind = local | halo | replicated | reduction,
  split_dim,
  left_halo,
  right_halo,
  boundary_policy
}
```

region ledger 至少记录：

```text
RegionVersion {
  buffer_id, logical_version, partition_scheme,
  owned_region[part], owned_version[part],
  ghost_region[part], ghost_source_version[part],
  canonical_regions
}
```

安全路线分两阶段：

1. `SplitHaloFullAllocation`：沿用每卡完整虚拟 allocation 和 global id，producer output
   的 own block 驻留，只复制相邻 halo；先取得通信复杂度收益，不改 kernel ABI；
2. `SplitRegionAllocation`：增加 accessor base/offset remap，让每卡只物理分配
   `owned + halo`。这能降低显存，但需要 compiler/runtime ABI 支持，应独立实现。

第一阶段已经能把连续 stencil chain 的每级数据维护从全量 replica/merge 降为边界
交换。只有 mapper 明确给出 halo width 且 producer/consumer partition scheme 一致时
才允许；间接索引、global reduction、未知 alias 一律 materialize 或回退 Single。

若要 overlap interior compute 与 halo copy，同一逻辑 part 还需拆成 interior launch 和
boundary launch；完成状态必须等两者都完成，但 profile 仍归并为一个逻辑 kernel mode。
不能仅把两次 launch 当作两个无依赖 kernel，否则 daemon 会错误发布未完成的 region
version。

### 3.3 跨 wait 的 residency

当前 resident version 只在同一用户 wait window 内跨 kernel 保留，window fence 前会
canonical materialize。它已消除长链中间级的 merge，但还不是跨 epoch 的 distributed
buffer。

跨 wait 保留要求把 version ledger 从 handler 的 window-local pending vector 提升到
buffer 生命周期，并在以下入口显式 materialize/失效：host accessor、buffer destructor、
interop/native handle、未知 queue/context consumer 和非兼容 mapper。`queue::wait()`
本身只要求执行完成，并不天然等价于 host 读；但在完成这些 buffer 生命周期 hook 前，
保留当前 fence materialization 是更安全的边界。

## 4. 下一步优先级

在本轮代码尚未构建和验证的前提下，不应继续叠加 multi-rank 或 halo ABI 改动。建议
顺序为：

1. 先验证本轮 profile journal 的解析、namespace 隔离、source 命中和 writer shutdown；
2. 用已有 benchmark 记录每种 estimate source 的覆盖率与误差，但不据此修改统一目标；
3. 需要论文覆盖多节点时，优先完成 dispatch lease + compute/data-ready ACK；
4. 论文主张 stencil 性能时，优先完成显式 `SplitHaloFullAllocation`，再考虑物理分区
   allocation；
5. compiler feature 冷启动与跨 wait residency分别作为模型面和数据面的后续扩展。

这个顺序保持已有正确性边界：profile 永不阻塞 ready queue，remote completion 不在
缺少 lease 时释放资源，Split 不在缺少访问证明时省略数据移动。
