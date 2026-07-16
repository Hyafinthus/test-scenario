# Runtime 调度演进：wait 语义、统一代价模型与三个创新点

## 结论

本轮实现遵守一个不可破坏的边界：用户代码中的 `queue::wait()` / `event::wait()` 仍然是语义 fence，runtime 必须等待该 fence 之前的相关工作完成后才能返回。profile 只是完成后的观测，不得再制造第二个等待条件，也不得要求所有 daemon/rank 为了交换 profile 同步到齐。

当前能够安全落地的修正包括：

1. handler 将“用户 fence 等待”和“读取 profiling timestamp”拆开；timestamp 查询本身不再调用 `event.wait()`。
2. daemon 的跨 rank profile 传播由同步 `Send/Recv` 改为 worker `MPI_Isend`、master `MPI_Iprobe` drain。只有进程退出时 join，运行期间不形成 profile barrier。
3. kernel profile key 增加稳定的 kernel symbol identity，避免 NDRange 和 accessor 形状相同的不同 kernel 共用错误样本。
4. monitor、profile、device capability 和 HEFT calendar 进入同一个批次调度上下文。monitor 只形成一次 service-time scale，不再同时作为虚构 available-time 和额外 penalty 重复计费。
5. 单 rank 多 GPU 模式下，在已确认的用户 fence 后重置历史节点的合成 finish-time；仍保留 producer placement 作为数据驻留位置。多 rank 暂不重置，因为缺少 remote completion ACK。
6. Split 默认开启，但仍经过 cold-probe、memory、layout 和 wide-DAG 可行域约束；可用 `SYCL_SNMD_ENABLE_SPLIT=0` 运行时回退。profiled Split 不再使用独立的固定 hysteresis。
7. dependent data transfer 不再只是加到 predecessor finish 上的标量 cost；HEFT 同时预留 source 与 destination device calendar，避免把排在繁忙 in-order queue 后的 copy 误判成可立即执行。
8. profile variance、样本数、估计来源和传输不确定性已进入统一风险目标；Single、迁移、Split、静态 HEFT 和 completion-driven queue 不再各自应用不同的 penalty/admission。monitor 仍只提供统一的 external-load scale。
9. 单 rank 已实现 completion-driven ready queue：handler 报告真实 event completion，
   daemon 释放 single/Split compute gang 后，再调用统一 candidate selector dispatch
   下一批 ready task；transfer endpoints 只在一个 dispatch wave 内按预测 copy 时间
   排序，不会锁到目标 kernel 完成。multi-rank 暂时保留静态 fallback。
10. profile observation 已异步持久化并进入分层预测：旧 exact、跨设备、同 identity
    跨 shape 和严格结构同类样本都能参与冷启动，同时以来源、年龄和迁移距离扩大
    uncertainty；profile 文件 I/O 不阻塞 completion-driven queue。

协议、状态机、当前 materializing Split 与后续 partition-resident/halo 设计详见 [`completion-driven-split-runtime.md`](completion-driven-split-runtime.md)。

`mc-result2.txt` 的逐 epoch 证据、修正后的验证判据和贡献成熟度见 [`mc-result2-analysis.md`](mc-result2-analysis.md)。

## 1. wait 的语义边界

### 必须相信用户代码

若程序写成：

```text
submit scenario 0..3 for epoch e
queue.wait()
submit scenario 0..3 for epoch e+1
```

那么第二个 epoch 的 kernel 在 `wait()` 返回前尚未提交。即使 scenario 0 已经完成，runtime 也不能凭空执行 scenario 0 的下一 epoch。此时某张 GPU 因其他 scenario 长尾而空闲，是应用 DAG 暴露程度造成的，不是 profile barrier 可以解决的。

要让 early-finished scenario 跨 epoch 前进，bench 必须显式暴露这种并行性，例如：

- 一次提交多个 epoch，用 event/accessor 依赖表达每个 scenario 自己的链；最后统一 wait；
- 每个 scenario 使用独立 queue/fence，由 host 分别推进；
- 或将多个独立病人/beam/parameter sample 放入同一个更宽的调度窗口。

runtime 可以在“已经提交的 ready kernel”和“其他应用的 ready kernel”中插空，但不能越过用户写下的 fence 去推测未来命令。

### handler 中允许等待什么

允许等待：

- 用户 fence 覆盖的每个实际 single-device event；
- Split 正确性路径中的 part completion 与最终 merge；
- host accessor 或跨 rank 数据传输要求的真实数据完成。

不允许额外等待：

- 为了读取 profile 再次等待同一 event；
- 为了让所有 rank 凑齐 profile 批次而等待最慢 rank；
- 为了更新 EWMA 而阻塞可执行 kernel。

## 2. 统一调度决策

旧实现中以下因素分别修改结果：monitor 初始化 available-time、`monitorPenalty` 再乘执行时间、profile 替换部分 cold estimate、通信模型单独参与 HEFT。这样同一个“GPU 正忙”的事实会被重复计算，也无法解释最终决策。

统一后的候选决策应只有一条主线：

```text
execution_cost(mode, device)
  = base_service_estimate(kernel, mode, device)
  × external_load_scale(device, scheduling_snapshot)

transfer_start(edge)
  = max(predecessor_finish,
        source_device_calendar,
        destination_device_calendar)

ready_time(candidate)
  = max(target_device_calendar,
        all transfer_end(edge))

EFT_mean(candidate) = ready_time + execution_cost.mean

risk(candidate)
  = EFT_mean(candidate)
  + beta × hypot(execution_cost.uncertainty,
                 transfer_uncertainty)
```

其中：

- `base_service_estimate` 优先使用同 kernel identity、同 device、同 `num_parts` 的 profile；无精确样本时才按设备能力缩放其他样本或使用 cold estimate；
- monitor 在一次 HEFT pass 开始时形成不可变快照，只贡献一个 external-load scale；
- 刚在该设备完成并上报的本程序 profile 会抑制短时间内陈旧 NVML busy 值，避免把已经结束的自身工作重复当作外部负载；
- communication cost 只由 producer placement、consumer placement、有效字节数和链路 profile 决定；
- communication reservation 同时推进 source/destination calendar，使迁移 copy 与两端已排队工作进入同一个资源模型；
- `gpu_available_time` 只表示本次 HEFT 已经放置的任务日历，不再编码 monitor busy。
- 精确 profile、能力缩放 profile、cold model 和 derived Split 都返回同一个 `CostEstimate`；能力缩放额外保留 target-model error，不能冒充精确样本。
- HEFT upward rank、逐节点 placement、co-located batch fallback 与 completion-driven 重选都使用同一个风险上界。风险分数等价时才依次偏好更少移动、更少设备和更早均值完成时间。

每个应用 daemon 线程持有独立的 HEFT/completion calendar、capability、memory
和 monitor-scale 快照；profile table 与原始 monitor 数据仍是加锁共享状态。
因此 completion loop 不会与另一应用 resize/覆盖 calendar，也不需要把一个全局锁
持有到数十秒 kernel 完成。不同应用争用同一物理 GPU 的影响由下一次 monitor
快照作为 external load 进入，而不是共享一个会跨应用污染 DAG 时间轴的合成 calendar。

## 3. 三个论文创新点及安全落地顺序

### 创新一：不确定性感知的冷启动与迁移学习

目标不是再增加一个经验系数，而是让每个 estimate 带来源和置信度：

```text
CostEstimate = {mean, variance, sample_count, source, timestamp}
```

当前统一风险目标为：

```text
risk(candidate) = mean_EFT + beta × predictive_uncertainty
candidate = argmin risk(candidate)
```

当前实现与后续验证顺序：

1. 已落地：稳定 kernel identity；profile 保存 EWMA、Welford mean/variance、min 和 sample count。
2. 已落地：精确 device/mode profile 将 Welford 观测标准差作为下一次运行的 aleatoric uncertainty，并叠加随样本数衰减的 cold-start prior；capability-scaled/cold 候选使用宽预测区间。观测抖动不会因为样本变多而错误地消失。
3. 已落地：显式 `CostEstimate {mean, uncertainty, samples, source}`；精确、缩放、cold、derived Split 共用该类型，monitor 同时缩放 mean/uncertainty，通信预测也提供 uncertainty。
4. 已落地：同一个 UCB 风险目标用于 HEFT rank、Single/Split placement、迁移、co-located fallback 与完成事件后的重选，删除 profiled Split 15% hysteresis 和 migration-only 置信门槛。
5. 已落地：rank 0 daemon 启动时加载 profile observation journal；完成线程只向有界队列 enqueue，后台线程持久化，磁盘写入不进入 completion/ready queue 临界路径。持久化样本有独立误差下限、时间衰减和 live/persisted 来源区分。
6. 已落地：预测回退层级为 `live exact -> persisted exact -> exact-key capability scaling -> same-identity shape transfer -> strict structural cohort -> analytical cold`。同 identity 以与 DAG 无关的 shape work proxy 缩放；跨 identity 只在访问结构严格同类时局部迁移，并保留更宽的模型不确定性。
7. 已落地：持久化记录保存采样时的 FP32/FP64 capability；跨设备/跨重启缩放使用记录能力，而不是错误读取本次进程中相同 rank/device 编号的当前能力。
8. 已落地：`SYCL_SNMD_PROFILE_NAMESPACE` 隔离不同应用/构建；正式论文运行应使用应用版本、git commit 或二进制哈希。未设置时为兼容模式 `default`，不能自动识别“kernel 名字不变但实现已修改”。
9. 待实验：按 estimate source 分桶检查 coverage/calibration，并验证 bounded cold probe 是否同时避免永久无样本与灾难性探索。
10. 待设计：由 compiler/runtime integration 提供指令混合、寄存器/共享内存、occupancy 与静态访存特征，替换当前跨 identity 的粗粒度 structural cohort。daemon 不应从 accessor metadata 虚构这些信息。

必须验证：两个形状完全相同但代码不同的 kernel 不共享样本；相同 kernel 跨 epoch 能命中；不同 `num_parts` 永不混桶。

### 创新二：profile feedback 与 completion-driven HEFT

真正的创新不是“HEFT 外面套 profile”，而是把调度状态从一次性静态表改为事件驱动状态机：

```text
SUBMITTED -> READY -> DISPATCHED -> COMPLETE -> PROFILED
```

`COMPLETE` 更新资源 calendar 和数据版本；`PROFILED` 只更新未来预测。两者必须解耦，因而缺 profile 不能阻止 ready task 调度。

当前已落地的基础：

- profile 跨 rank 非 collective 传播；
- 单 rank 在用户 fence 后 rebase 已完成历史 DAG 的合成时间；
- profile key 在 handler 清理 command group 前被捕获，不依赖延迟对象生命周期。
- HEFT 的 dependent transfer 同时预留 source/destination device calendar，迁移不再被当作不占资源的常数 penalty。
- 单 rank 的 `PENDING/READY/DISPATCHED/COMPLETE` 状态机、completion/dispatch 消息和稳定 per-device in-order queue；一个设备只有在真实 completion 后才重新进入 candidate 集合。
- Split 以 gang reservation 进入同一 ready queue，完成确认覆盖所有 part 与 canonical materialization。

单 rank rolling dispatch 已落地；尚未安全实现的是 multi-rank completion-driven HEFT。后续顺序：

1. 增加 remote `COMPLETE(kernel, rank, device, version)` ACK；
2. master 对 remote in-flight reservation 增加超时和 daemon failure recovery；
3. completion 与跨 rank data-ready 分离，避免计算完成但 MPI/共享内存尚未 materialize；
4. 完成后再允许 multi-rank rolling rebase。

每一步都要保留旧的 batch-static 路径作为 fallback，并比较同一 DAG 的依赖闭包与最终 checksum。

### 创新三：partition-resident Split dataflow

达到 stencil/Celerity 类性能的关键不是让更多 kernel 被 Split，而是避免每次 Split 都构造全量 snapshot 并在 batch 末合并全量结果。需要显式的区域版本状态：

```text
BufferVersion {
  logical_version,
  valid_regions[device],
  complete_replica_devices,
  last_writer,
  dirty_regions
}
```

只有当 consumer 的读区域被目标设备上的有效 region 完全覆盖时，才允许复用；不能以“该设备曾有 allocation”替代版本证明。

安全演进：

1. 当前：Split 默认开启，但仅允许 dim0 可整除且 writable region 连续的 kernel；gang reservation 与 completion queue 共同管理全部 part devices。
2. 已实现 wait-window 内的 read-only full replica cache，并在任意写访问前失效；writable output 仍 canonical materialize。
3. 再实现一对一 partition producer/consumer：相同 partition 直接链式执行，直到真实 host access 才 materialize。
4. 再加入 stencil halo region 与邻接 copy。
5. 最后才允许动态 repartition、跨设备完整 replica 和跨 rank partition。

对每一阶段必须有 host reference、随机尺寸、不可整除尺寸、多 accessor element size、read/write/atomic、异常传播和连续多 window 测试。任一版本证明失败时回退 single-device，不得猜测。

## 4. radio bench 的直接含义

`mc-result2.txt` 当前每个 epoch 是 8 个互不依赖 scenario kernel，DAG 宽度为 8；较早的 4-scenario 配置宽度为 4。两者在四张 A6000 上都首先应使用 whole-kernel task parallelism；几乎没有 Split 是合理结果，因为已有足够独立、长时间 kernel 占满四张卡，Split 一个 scenario 会抢占本可运行另一个 scenario 的 GPU，还会增加输入复制和结果 merge。

旧 profile key 不含 kernel identity，四个形状完全相同的 scenario 会落入同一 key。若四个场景计算路径或设备运行时间略有不同，EWMA 会把它们混成一个均值，并可能导致下一 epoch 换卡。新 key 能区分不同 kernel symbol；如果四个 scenario 实际由同一个模板 kernel symbol 实例产生，它们仍会共享 service profile，此时数据驻留和 producer placement 应负责 affinity，而不是把场景编号硬编码进 profile。

单 rank rebase 修复了另一个换卡来源：上一 epoch 的合成 `finish_time` 不再被当作下一 epoch 仍未完成的占用；历史节点只保留数据 producer/placement。因真实 A6000 之间仍可能有长尾，HEFT 可以基于各卡 EWMA 做重新平衡，但迁移必须同时支付实际数据移动成本。

## 5. 验证矩阵

### 构建一致性

本轮修改了 handler-daemon 协议，新增 `kernel_identity` 字段。必须同时重编译 runtime/handler 与 `sycl-daemon`，不能混用旧二进制。

### 默认路径

1. radio 宽 DAG 在 Split 默认开启时仍应全部为 `num_parts=1`；另用 `SYCL_SNMD_ENABLE_SPLIT=0` 做 single-only 消融。
2. 比较 native、修改前 runtime、修改后 runtime 的 checksum 与 `VERIFY passed=1`。
3. 运行至少 20 epochs，记录每个 `(kernel identity, epoch, device)` 的 placement、raw profile、service scale、comm bytes 和 completion time。
4. 检查第二个 epoch 起 `AvailableTime=0`，刚完成设备对应 `FreshProfile=1` 时不重复施加 NVML penalty。

### wait 正确性

提交多个互不依赖、时长不同且位于不同 GPU 的 kernel；在 `queue.wait()` 返回后立即读取每个输出。任何未完成、旧值或只等待最后提交 event 的现象都判失败。

### 非阻塞 profile

多 rank 上人为让一个 rank kernel 多运行数秒。先完成 rank 的 daemon 应能继续处理消息；profile 路径中不应再出现每轮固定次数的 `PROFILE_LEN_TAG` recv。退出时允许 profile flush join。

### Split 灰度

Split 现在默认开启；可用以下命令做 fallback 消融：

```bash
SYCL_SNMD_ENABLE_SPLIT=0 mpirun -n 1 sycl-daemon
SYCL_SNMD_COMPLETION_QUEUE=0 mpirun -n 1 sycl-daemon
```

先用单 kernel、dim0 连续写、可整除规模验证；radio 宽 DAG 不是证明 Split 有效的正例。持续 partition 的性能正例应使用窄 DAG、相同 partition 连续生产消费、低 halo/低 materialization 频率的 stencil 或 block pipeline。

## 6. 当前能力边界

- 已消除由 profiling 自身造成的 event 二次等待和跨 rank profile barrier。
- 未绕过、也不应绕过用户写下的 epoch barrier。
- 单 rank 多 GPU 有完成后的 calendar rebase；multi-rank 必须等 completion ACK 后再启用。
- 当前 scheduler 已统一 monitor/profile/communication/HEFT/Split 的估计类型和风险目标；source/destination transfer reservation 已进入相同目标。profile 已可跨 daemon 重启积累并进行受不确定度约束的 shape/设备迁移，但仍需 A6000 重跑验证预测区间与实际执行一致。
- handler trace 已记录每次 `resubmit` 的 host duration，用于确认换卡时是否在跨 context 数据准备中阻塞；该 trace 只在 handler trace 开启时输出。
- 当前 Split 已默认启用并使用 gang reservation、精确 part 数 profile、read-only replica reuse，以及 compute/materialization 分项计时；writable output 仍走 canonical materialization。达到 stencil 性能仍需要显式 region-version、partition contract 与 halo persistence，不能靠推断 accessor/NDRange 对应关系跳过 merge。
