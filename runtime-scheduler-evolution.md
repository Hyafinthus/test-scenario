# Runtime 调度演进：wait 语义、统一代价模型与三个创新点

## 结论

本轮实现遵守一个不可破坏的边界：用户代码中的 `queue::wait()` / `event::wait()` 仍然是语义 fence，runtime 必须等待该 fence 之前的相关工作完成后才能返回。profile 只是完成后的观测，不得再制造第二个等待条件，也不得要求所有 daemon/rank 为了交换 profile 同步到齐。

当前能够安全落地的修正包括：

1. handler 将“用户 fence 等待”和“读取 profiling timestamp”拆开；timestamp 查询本身不再调用 `event.wait()`。
2. daemon 的跨 rank profile 传播由同步 `Send/Recv` 改为 worker `MPI_Isend`、master `MPI_Iprobe` drain。只有进程退出时 join，运行期间不形成 profile barrier。
3. kernel profile key 增加稳定的 kernel symbol identity，避免 NDRange 和 accessor 形状相同的不同 kernel 共用错误样本。
4. monitor、profile、device capability 和 HEFT calendar 进入同一个批次调度上下文。monitor 只形成一次 service-time scale，不再同时作为虚构 available-time 和额外 penalty 重复计费。
5. 单 rank 多 GPU 模式下，在已确认的用户 fence 后重置历史节点的合成 finish-time；仍保留 producer placement 作为数据驻留位置。多 rank 暂不重置，因为缺少 remote completion ACK。
6. Split 默认关闭，但可通过 `SYCL_SNMD_ENABLE_SPLIT=1` 显式启用，便于逐项正确性验证，不再需要反复重编译实验开关。

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

ready_time(candidate)
  = max(device_calendar,
        predecessor_finish + placement_aware_communication)

EFT(candidate) = ready_time + execution_cost
```

其中：

- `base_service_estimate` 优先使用同 kernel identity、同 device、同 `num_parts` 的 profile；无精确样本时才按设备能力缩放其他样本或使用 cold estimate；
- monitor 在一次 HEFT pass 开始时形成不可变快照，只贡献一个 external-load scale；
- 刚在该设备完成并上报的本程序 profile 会抑制短时间内陈旧 NVML busy 值，避免把已经结束的自身工作重复当作外部负载；
- communication cost 只由 producer placement、consumer placement、有效字节数和链路 profile 决定；
- `gpu_available_time` 只表示本次 HEFT 已经放置的任务日历，不再编码 monitor busy。

多个应用 daemon 线程仍共享旧版全局 HEFT 容器，因此本轮给整个元数据决策加互斥边界，防止两个应用同时 resize/覆盖 calendar。锁不覆盖 kernel 执行、event wait、profile 传输或共享内存数据传输。

## 3. 三个论文创新点及安全落地顺序

### 创新一：不确定性感知的冷启动与迁移学习

目标不是再增加一个经验系数，而是让每个 estimate 带来源和置信度：

```text
CostEstimate = {mean, variance, sample_count, source, timestamp}
```

候选排序使用风险调整成本：

```text
risk_cost = mean + beta × standard_error
```

建议分阶段实现：

1. 已落地：稳定 kernel identity；profile 保存 EWMA、Welford mean/variance、min 和 sample count。
2. shadow 阶段：同时记录旧决策与 risk-aware 决策，不改变 placement；以 radio 的重复 epoch 做离线 replay。
3. 只在 `sample_count >= N` 或候选预测收益大于置信区间时启用新决策。
4. 设备间迁移只使用 capability ratio 作为先验；观测到目标设备样本后逐步降低先验权重。

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

尚未安全实现的是同一 window 内的 rolling HEFT。它需要新增明确的 completion 消息和 daemon 持久 DAG，不能用后台线程直接读取现有 `CG` 指针来模拟。建议顺序：

1. feature flag 下只发送 `COMPLETE(kernel, device, version)`，daemon 记录但不改变调度；
2. 对单 rank、non-Split kernel 启用 ready-queue 补位；
3. 加入 Split part/merge completion；
4. 最后加入 multi-rank ACK 和失败恢复，再允许跨 rank rebase。

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

1. 当前：Split 默认关闭；显式 opt-in；仅允许 dim0 可整除且 writable region 连续的 kernel。
2. 先实现 read-only full replica 的版本标记，不改变 writable merge。
3. 再实现一对一 partition producer/consumer：相同 partition 直接链式执行，直到真实 host access 才 materialize。
4. 再加入 stencil halo region 与邻接 copy。
5. 最后才允许动态 repartition、跨设备完整 replica 和跨 rank partition。

对每一阶段必须有 host reference、随机尺寸、不可整除尺寸、多 accessor element size、read/write/atomic、异常传播和连续多 window 测试。任一版本证明失败时回退 single-device，不得猜测。

## 4. radio bench 的直接含义

radio 当前每个 epoch 是 4 个互不依赖 scenario kernel，DAG 宽度正好为 4。四张 A6000 能提升，是 whole-kernel task parallelism 生效；几乎没有 Split 是合理结果，因为已经有四个独立、长时间 kernel 可以占满四张卡，Split 一个 scenario 会抢占本可运行另一个 scenario 的 GPU，还会增加输入复制和结果 merge。

旧 profile key 不含 kernel identity，四个形状完全相同的 scenario 会落入同一 key。若四个场景计算路径或设备运行时间略有不同，EWMA 会把它们混成一个均值，并可能导致下一 epoch 换卡。新 key 能区分不同 kernel symbol；如果四个 scenario 实际由同一个模板 kernel symbol 实例产生，它们仍会共享 service profile，此时数据驻留和 producer placement 应负责 affinity，而不是把场景编号硬编码进 profile。

单 rank rebase 修复了另一个换卡来源：上一 epoch 的合成 `finish_time` 不再被当作下一 epoch 仍未完成的占用；历史节点只保留数据 producer/placement。因真实 A6000 之间仍可能有长尾，HEFT 可以基于各卡 EWMA 做重新平衡，但迁移必须同时支付实际数据移动成本。

## 5. 验证矩阵

### 构建一致性

本轮修改了 handler-daemon 协议，新增 `kernel_identity` 字段。必须同时重编译 runtime/handler 与 `sycl-daemon`，不能混用旧二进制。

### 默认路径

1. 不设置 `SYCL_SNMD_ENABLE_SPLIT`，确认 radio 的 16 个 kernel 全部为 `num_parts=1`。
2. 比较 native、修改前 runtime、修改后 runtime 的 checksum 与 `VERIFY passed=1`。
3. 运行至少 20 epochs，记录每个 `(kernel identity, epoch, device)` 的 placement、raw profile、service scale、comm bytes 和 completion time。
4. 检查第二个 epoch 起 `AvailableTime=0`，刚完成设备对应 `FreshProfile=1` 时不重复施加 NVML penalty。

### wait 正确性

提交多个互不依赖、时长不同且位于不同 GPU 的 kernel；在 `queue.wait()` 返回后立即读取每个输出。任何未完成、旧值或只等待最后提交 event 的现象都判失败。

### 非阻塞 profile

多 rank 上人为让一个 rank kernel 多运行数秒。先完成 rank 的 daemon 应能继续处理消息；profile 路径中不应再出现每轮固定次数的 `PROFILE_LEN_TAG` recv。退出时允许 profile flush join。

### Split 灰度

只有在默认路径通过后才运行：

```bash
SYCL_SNMD_ENABLE_SPLIT=1 mpirun -n 1 sycl-daemon
```

先用单 kernel、dim0 连续写、可整除规模验证；radio 宽 DAG 不是证明 Split 有效的正例。持续 partition 的性能正例应使用窄 DAG、相同 partition 连续生产消费、低 halo/低 materialization 频率的 stencil 或 block pipeline。

## 6. 当前能力边界

- 已消除由 profiling 自身造成的 event 二次等待和跨 rank profile barrier。
- 未绕过、也不应绕过用户写下的 epoch barrier。
- 单 rank 多 GPU 有完成后的 calendar rebase；multi-rank 必须等 completion ACK 后再启用。
- 当前 scheduler 已统一现有 monitor/profile/communication/HEFT 的作用位置，但 variance-aware admission 仍处于数据准备阶段，尚未改变 placement。
- 当前 Split 仍是 snapshot/merge 路径，只有显式 opt-in；达到 stencil 性能需要真实 region-version 与 partition persistence，不能只放宽 Split admission。
