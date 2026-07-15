# Completion-driven 统一调度与 Split 演进设计

## 1. 不变量

本设计保持以下语义边界：

1. 用户 `queue::wait()` / `event::wait()` 仍是最终 fence；返回前，窗口内所有 single kernel、Split part 和必要的 canonical materialization 必须完成。
2. `COMPLETE` 与 `PROFILED` 分离。完成消息即使没有合法 duration，也必须释放设备；profile 只能更新未来预测，不能成为 ready task 的等待条件。
3. single、Split、迁移、monitor、通信和数据局部性只经过同一个 candidate admission；handler queue 不得再做第二套设备选择。
4. 当前 completion-driven 协议只在 `daemon_size=1`、`onrun_size=1` 时启用。multi-rank 保留 batch-static fallback，直到具备 remote completion ACK、超时和失败恢复。
5. Split 只有在写区域满足当前 dim0 连续分块证明时才可执行；跨 kernel
   驻留还要求用户逐 accessor 声明 partition-local contract。任何证明失败均回退
   canonical materialization 或 single-device，runtime 不猜测 kernel 内索引。

## 2. 统一候选决策

每个 ready node 生成以下候选：

```text
Single(rank, device)
Split(rank, {device_0 ... device_n-1})
```

统一计算：

```text
execution = profile_or_cold_estimate
          * monitor_snapshot_service_scale

transfer_start = max(predecessor_complete,
                     source_device_calendar,
                     destination_device_calendar)

candidate_mean_EFT = max(target_calendar, all_transfer_end) + execution.mean

candidate_risk = candidate_mean_EFT
               + beta * hypot(execution.uncertainty,
                              transfer.uncertainty)
```

四种 service estimate 使用同一个类型：

```text
CostEstimate { mean, uncertainty, samples, source }
source in {exact-profile, scaled-profile, cold, derived-split}
```

cold path 现在使用 accessor 的实际 access range，而不是把 backing buffer 全尺寸都
当作计算工作量；kernel identity、NDRange、访问模式、元素宽度、读写规模、设备
FP32/FP64 capability 和 monitor service scale 共同构成先验。这里的“统一”已在实现
上闭环，但“跨 workload 冷启动精度”仍是待校准的实验命题：当前还没有指令混合、
occupancy/cache 特征，也没有把 profile table 持久化到 daemon 重启之后。

`selectUnifiedTaskCandidate()` 是 batch HEFT 和 completion dispatcher 的共同入口。profiled Split 不再使用固定 15% hysteresis，迁移也不再使用另一套 pairwise 门槛；两者与 Single 一起最小化 `candidate_risk`。co-located fallback 只有在同一 batch risk makespan 更小时才能覆盖逐节点结果。queue 只执行 daemon 返回的决定，不重新读取 NVML、不增加独立 penalty，也不自行改卡。

## 3. Completion-driven ready queue

### 3.1 状态机

```text
PENDING --all in-window predecessors complete--> READY
READY   --candidate admitted and resources reserved--> DISPATCHED
DISPATCHED --handler event(s) complete--> COMPLETE
COMPLETE --duration available--> PROFILED
```

`PROFILED` 不是资源状态；一个 node 可以已经 `COMPLETE`、设备已经释放，但 profile 因后端不支持 profiling 而永远缺失。

### 3.2 daemon 行为

1. 先用完整 DAG 计算 HEFT upward rank，作为 ready task 优先级。
2. 清空本窗口的预测 calendar；历史 producer placement 仍保留，用于数据局部性和 transfer source。
3. 从高 rank 的 READY node 开始调用统一 candidate selector。
4. single 跨 completion 占用目标 device；Split 以 gang 方式同时占用所有 part
   devices。数据移动在同一个 dispatch wave 内预留 source/destination endpoints，
   wave 发出后释放短期 transfer calendar；不能把 source 锁到目标计算完成。
5. 将所有当前可并行 dispatch 的任务作为一个 `DISPATCH_BATCH_V1` 发给 handler。
6. 收到 `COMPLETION_BATCH_V1` 后释放该 kernel 的全部 reservation，更新 profile（若 duration 非零），再选择下一批 READY node。
7. 只有全部 node 都为 `COMPLETE` 时发送 `window_complete=1`。

协议一旦发送首个动态 batch 就不允许静默降级为静态消息；后续出现未知、重复
completion 或 ready-queue deadlock 时，daemon 发送 `window_failed=1`，handler
抛出 runtime error。这样不会让用户 `wait()` 在部分 kernel 未完成时错误返回。
若在首个动态 batch 发送前即发现无法启动，daemon 会恢复此前 HEFT 的 node
placement 与 calendar，再使用已序列化的 batch-static fallback；不会只恢复消息而
留下被清空的历史 producer placement。
反向路径同样存在：submission、event wait 或 Split materialization 抛错时，handler
先发送失败 completion，使 daemon 放弃当前 reservation 状态，再把原始 SYCL
异常传播给用户。

busy compute device 在动态 calendar 中表示为 infinity。因此一个 task 不会提前排在
该设备的第二个计算位置；只有真实 completion 到达后，设备才重新参与计算候选。
transfer 的有限 reservation 只用于同一 dispatch wave 的冲突排序，实际顺序由稳定
in-order queue 保持。这既消除了预排 queue 的长尾空转，也避免把一次毫秒级 D2D
copy 的 source GPU 错锁到目标数十秒 kernel 完成。

calendar 与该 daemon 线程绑定，不是进程全局可变容器。多个应用可各自等待完成和
补发任务，不会互相覆盖 calendar，也不会为了保护元数据而持有一个贯穿 kernel
执行期的全局锁；共享 profile/monitor 只以受保护快照进入候选代价。

### 3.3 handler queue 行为

- 每个设备复用一个 private-context、in-order、profiling queue。
- handler 只提交当前 dispatch batch，不再一次性提交整个 wait window。
- daemon 在同一个已批准的 dispatch wave 内先发送零移动 kernel，再发送需要迁移的
  kernel；这是 submission ordering，不改变 placement。即使一次合法的跨 context
  准备在 `resubmit()` 内短暂阻塞，也不会先挡住本可立即填充其他 GPU 的 local work。
- event 状态以约 1 ms 周期检查；已完成 single event 调用 `wait()` 传播异步异常并读取 profiling timestamp。
- Split 必须等所有 part event 完成才向 daemon 报告完成。普通 Split 在报告前完成
  canonical merge；resident Split 报告的是“所有 partition ready”，版本仍留在原
  device。daemon 用独立的 `persistent_split` mode 区分两种完成语义。
- 多个同时完成的 kernel 合并为一个 completion batch，减少消息次数。

## 4. 当前启用的 Split：materialize 与 resident 两种 gang mode

Split 默认开启，可用以下方式回退：

```bash
SYCL_SNMD_ENABLE_SPLIT=0
SYCL_SNMD_COMPLETION_QUEUE=0
```

Split admission 同时要求：

- `num_parts` 为 2 或 4，global dim0 和 writable accessor dim0 可整除；
- writable region 能表示为连续 row block；
- device memory 足够；
- 宽 DAG 已足够填满 GPU 时不抢占 task parallelism；
- 已有 single/Split profile 时按同一个风险上界比较，样本抖动、模型来源和 monitor scale 都显式参与；
- 无 Split profile 的 cold probe 只允许 single estimate 至少为 500000 cost units（约 5 s），并且预测收益至少 30%。

所有未声明 partition contract 的程序仍执行 `SplitMaterialize`：

1. Split 是 compute gang reservation，而不是把多个 part 当作互不相关的 kernel；
   任一 part device 正在计算就不 dispatch，同 wave 的 transfer endpoints 按有限
   copy calendar 串行 admission。
2. materializing Split profile 记录 `parts completion wall time + actual materialization duration`，不再把“等待到什么时候才需要 merge”的空闲区间误算为 kernel cost。
3. 完整输出在首个 Split device canonicalize；daemon 与 handler 使用同一个 canonical source。
4. 同一 wait window 内，已复制到某设备的只读输入形成 replica cache；后续 Split 在同 device/context 上复用，任何写访问在提交前使该 buffer 的 cache 失效。
5. direct D2D merge 失败时仍回退 D2H→H2D；正确性优先于保留 Split。

显式 contract 的链则可选择 `SplitResident`。profile key 同时包含完整 access range、
offset、sub-buffer 和 partition-local 标志；profile mode 另含
`persistent_split`，因此 resident 的“准备 + part wall time”不会与 materializing
的“准备 + part wall time + merge”混合。若 consumer 不兼容，merge 作为依赖边
transfer cost 计一次，不会同时重复进入 producer profile。

## 5. 已实现的 partition-resident Split

不能仅凭“accessor range 等于 buffer range”推断 kernel part 只访问自己的 NDRange
对应数据。本轮采用显式、逐 accessor 的实验 contract：

```cpp
queue.submit([&](sycl::handler &h) {
  auto in = a.get_access<sycl::access::mode::read>(h);
  auto out = b.get_access<sycl::access::mode::discard_write>(h);
  h.ext_snmd_partition_local(in);
  h.ext_snmd_partition_local(out);
  h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
    out[i] = f(in[i]);
  });
});
```

声明的含义是：每个 Split part 只访问与其 dim0 work-item block 对应的 buffer row
block。系数表、全局归约、间接索引、halo/global read 不得标记。handler 再验证：

- 所有 writable accessor 都有 contract，且没有 atomic；
- accessor 非 sub-buffer、offset 全零、覆盖完整 dim0 和 trailing dimensions；
- dim0/global range 可被 parts 整除，且 ordinary Split 连续 row-block 证明也成立；
- producer/consumer 的 parts、device order 和 block boundary 完全相同；
- 当前只在单 daemon/rank 开启 resident mode。

handler 的 `PendingOfflineSplitMerge` 现在同时承担 window-local version ledger：记录
producer、part events、ordered device set、written buffers 和已被后继覆盖的旧版本。
producer completion 只等待所有 part，不 merge；兼容 consumer 在同一组稳定
in-order queues 上直接消费对应 device allocation。首次输入若也声明
partition-local，只向每卡复制其 own block，不再给每卡完整 snapshot。新版本提交后，
旧版本按 buffer 标为 superseded，最终物化时不会把 stale part 写回 canonical。
同一 scheme 上的 partition-local RAW、WAR 和 WAW 都由这些 in-order queues 保序；
若后继不消费 producer 输出但覆盖 producer 的分区只读输入，runtime 也会识别这条
反依赖，只有 scheme 兼容时才能不 gather。

当前仍在每张 part device 上保留完整虚拟 buffer allocation，保证原 kernel 按 global
id 索引时无需指针重写；只是每张卡仅有 own block 的有效版本。daemon 因而按完整
allocation 做 memory admission，不能把 writable storage 错算成 `bytes/parts`。未来若
要降低显存占用，需要 accessor base/offset remap 或真正的 distributed buffer ABI，
这是独立于“消除每级搬运/merge”的下一项数据布局工作。

daemon 与 handler 使用同一组条件。daemon 对 resident/materializing profile 分 mode，
compatible edge 的 transfer 为零；incompatible edge 先 gather 到 producer ordered
device set 的首卡（与 handler canonical merge 一致），再估计到 consumer 的完整
replication，并在
completion dispatch wave 中预留隐藏的 source/target endpoints，避免一边运行
compatible successor、一边错误并发 materialize 同一 resident source。

物化发生在三种边界：不兼容 kernel、single/host 路径需要完整版本、或用户
`wait()` 结束当前 window。wait 返回前仍恢复完整 canonical 可观察语义；跨 wait
不会错误沿用 partition residency。没有 contract 的现有程序行为不变。

### 5.1 尚未实现的下一层：SplitHalo / partial materialization

当前 contract 是严格 partition-local，因此 pointwise transform、独立粒子/历史、
逐行 operator chain 可以直接使用。真正 stencil 还需显式 halo width/mapper：

```text
interior_region(device_i) stays local
right_halo(device_i) -> left_halo(device_i+1)
left_halo(device_i)  -> right_halo(device_i-1)
```

后续才应加入 `SplitHalo`、region read-set 和按需 partial materialization。边界宽度
必须来自 mapper/contract，不能由 runtime 猜测；在此之前 stencil accessor 不应
谎报为 partition-local。

## 6. 验证顺序

### 当前实现

1. `SYCL_SNMD_ENABLE_SPLIT=0`、completion queue 开：验证 single-only checksum、每卡最多一个 in-flight kernel、completion dispatch/done 数相同。
2. completion queue 关：验证旧 batch-static fallback checksum。
3. Split 开、宽 DAG：应保持 `split=0`，证明 task parallel guard。
4. Split 开、窄 DAG：验证 2-way cold probe、canonical checksum、materialization bytes 和 profile。
5. 2-way/4-way、奇数和不可整除尺寸、多 read-only input、write/read_write、D2D 失败回退。
6. 至少重复 20 windows，比较 migration、GPU idle gap、completion round-trip、merge bytes 和总时间。
7. resident chain：验证中间 kernel 的 merge bytes 为零、
   `resident_reused_bytes>0`、旧版本 supersede、最后 fence checksum 正确。
8. contract 反例：offset/sub-buffer、atomic、未标记 writable、设备顺序变化都必须
   自动回退，不得跳过 materialization。

radio 建议先做一宽一窄两个直接对照：

```bash
# 宽 DAG：4 个 scenario 应走 4 个 whole-kernel placement，Split=0
./mc_radiotherapy_scenarios_sycl_timer --work-items 4194304 \
  --scenarios 4 --epochs 4

# 窄 DAG：第 1 epoch 建立 single profile；足够长时后续 epoch 可 probe 2-way Split
./mc_radiotherapy_scenarios_sycl_timer --work-items 4194304 \
  --scenarios 1 --epochs 4 --split-parts 2
```

两组都必须与原生 checksum 一致。窄 DAG 若仍不 Split，先从 trace 区分
`min_single`、30% cold gain、layout/memory guard 和实际 profile，而不是直接降低所有
阈值；这能避免为了制造 Split 数量重新引入 Reactive 的灾难性全量复制。

resident 专用链：

```bash
./persistent_split_chain_sycl_timer --items 4194304 \
  --stages 4 --epochs 2 --inner-iters 4096
```

该 bench 每个 window 是宽度 1 的四级 ping-pong DAG。四个 accessor 都是严格
逐 item 映射，适合验证 `SplitResident`；`--wait-each-kernel 1` 刻意切断 residency，
作为每级 materialize 对照。每增加一种新 contract，仍必须加入错误 contract 拒绝、
host reference、随机尺寸、halo 边界、alias/atomic fallback 和跨 window invalidation。

## 7. 诊断输出

开启 daemon/handler trace 后应出现：

```text
CompletionQueue: dispatch_order ... kernel ... proc ... parts ...
CompletionQueue: complete kernel ... duration_ns ...
CompletionQueue: wait ... complete kernels ...
algorithmHEFT: ... cost_source ... profile_samples ... risk_score ...
Offline profiling split wall ... compute_wall_ns ... materialization_ns ...
Split step3 reuse read-only replica
Split step3 reuse resident partitioned version from kernel_count: ...
Split completion retained resident partitions for kernel_count: ...
```

`analyze_mc_runtime.py` 在存在 completion trace 时优先使用实际动态 placement，而不是 shadow HEFT 的初始静态 assignment。
