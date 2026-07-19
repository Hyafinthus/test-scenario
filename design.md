# 系统核心设计：不确定性感知、完成驱动的自适应 DAG 调度与分区驻留 Split

## 摘要

本系统面对的核心问题不是简单地“把 SYCL kernel 放到多张 GPU 上”，而是：当用户在
一个同步窗口内提交了具有依赖关系的一组 kernel 时，runtime 如何在运行前信息不完整、
设备性能和负载会变化、数据移动可能淹没计算收益的条件下，联合决定：

1. 哪些独立 kernel 应分别放到不同设备，利用 kernel-level task parallelism；
2. 哪个超大 kernel 值得占用多张设备执行 Split；
3. kernel 应放在哪组设备，是否值得为此迁移数据；
4. Split 后的数据应立即合并，还是以分区形式继续驻留供后继 kernel 消费；
5. 当预测与实际运行不一致时，如何尽快利用真实完成状态修正后续决策。

现有实现围绕这一问题形成了三个相互依赖的机制：

- **不确定性感知的分层性能预测**：把 analytical cold model、当前运行 profile、跨
  daemon 持久化 profile、跨设备缩放和跨 shape 迁移统一为带来源与预测不确定度的
  `CostEstimate`；
- **完成状态驱动的负反馈 DAG 调度**：HEFT 只提供全局 DAG 优先级，实际 placement
  由每次真实 completion 后重新执行的统一候选选择器决定，避免一次预测误差固化为
  整个窗口的长尾；
- **显式访问合同证明的 partition-resident Split**：调度器不仅选择 Split，还区分
  materializing Split 与 resident Split；只有用户逐 accessor 声明且 runtime 再次
  验证的分区局部访问，才允许 writable partition 跨 kernel 留在原设备。

三者组合后的完整故事是：**runtime 从用户 wait 窗口获得一个全局 DAG，用带不确定度
的模型联合比较 task placement、数据移动与 Split；handler 执行被批准的 dispatch wave；
真实 completion 释放资源并触发下一次决策；执行产生的 profile 和分区版本又分别更新
未来的计算代价与数据移动代价。** 这不是在 HEFT 外侧依次叠加 monitor、profile 和
Split heuristic，而是一个由观测、决策、执行、完成和数据状态构成的闭环。

当前最完整、最诚实的论文范围是：

> 面向单节点多 GPU、wait-delimited SYCL DAG 的不确定性感知自适应调度；runtime 在
> task-level parallelism 与 contract-proven intra-kernel Split 之间联合选择，并通过
> completion feedback 和 partition residency 减少预测长尾与重复数据物化。

如果把范围扩大到通用多节点 completion scheduling 或通用 stencil/分布式 buffer，
仍有明确的机制未实现，本文最后逐项列出。

---

## 1. 研究动机

### 1.1 单设备 SYCL 与静态多设备分配之间的空缺

原生单设备 SYCL 的优势是控制面短、数据状态简单、queue 行为直接；它的不足是无法
利用一个节点中的多张 GPU。一个多设备 runtime 若只是轮转设备或平均切分 NDRange，
则很容易在以下场景中比单设备更慢：

- kernel 很小，调度、重提交、profile 和数据准备开销无法摊薄；
- DAG 已经有足够的独立 kernel，继续 Split 只会让一个 task 抢占本来可执行其他 task
  的 GPU；
- kernel 虽然计算量大，但 Split 要给每张 GPU 复制完整输入并在每级合并输出；
- 四张同型号 GPU 仍会因瞬时负载、queue 排队、数据位置和实际 kernel 抖动产生长尾；
- 运行前模型不准确，静态调度器一次选错设备后，后续 kernel 已经被预排到错误 queue，
  即使其他 GPU 提前空闲也无法插空。

因此，问题不是“是否支持多 GPU”，而是“何时选择哪一种并行性，以及预测错了以后
能否纠正”。

### 1.2 应用本身同时暴露两种并行性

一个 wait 窗口内通常同时存在：

- **逻辑并行或 task parallelism**：多个互不依赖的 scenario、replica、beam、sample
  或 pipeline 可以作为完整 kernel 分别运行；
- **数据并行或 Split parallelism**：单个极重 kernel 的 NDRange 可以划分给多张 GPU。

这两者争用相同设备。假设四张 GPU 上有四个独立且足够大的 scenario kernel，正确
选择通常是一卡一个 whole kernel；若只有一个数十秒 kernel，才可能值得 2-way/4-way
Split。Split 因而不是 handler 层的局部执行技巧，而是与整个 DAG 宽度、后继依赖、
数据位置、设备负载和预测风险共同决定的全局资源选择。

### 1.3 HEFT 的问题不是没有 DAG，而是决策只发生一次

HEFT 能根据 upward rank、设备能力和通信代价为 DAG 建立一个合理静态计划，但传统
做法默认预测完成时间足以代表实际完成时间。一旦出现以下误差，静态计划不会自修正：

- 某一 GPU 的 kernel 比 EWMA 或 cold estimate 慢；
- monitor 恰好采到本程序上一轮的 100% 利用率，并把已经结束的工作再次视为未来负载；
- 数据准备在 host `resubmit()` 中阻塞，实际 queue start 晚于计划；
- Split 某一 part 或 canonical merge 形成长尾；
- 预测的迁移只是一项标量 cost，实际上 copy 被排在繁忙的 source/destination queue 后。

这会产生“某张卡已空闲，但下一 kernel 仍被锁在另一张繁忙卡后面”的现象。解决它
需要把完成状态变成调度输入，而不是仅在整个 epoch 结束后更新一张 profile 表。

### 1.4 profile 不是天然可用的调度知识

简单的 profile feedback 仍有四个问题：

1. daemon 重启后样本消失，新的运行仍从粗糙 cold model 开始；
2. 相同 kernel 的其他 shape 或其他设备已有样本时，exact-key lookup 仍可能完全不用；
3. 历史样本、跨设备缩放和 analytical model 的可信度不同，不能都当成一个确定值；
4. 若 profile 收集或跨 rank 汇总形成 barrier，它本身会阻止 ready work 插空。

因此，“有 profile”不等于“profile 已融入调度”。预测必须显式表达来源、样本数、
波动和迁移误差，调度器也必须以同一种风险语义消费它。

### 1.5 Split 的瓶颈是数据生命周期，不只是 kernel 加速比

朴素 Split 的执行流程通常是：

```text
复制输入到每张 GPU
  -> 每张 GPU 执行一个 part
  -> 将所有 writable part 合并成完整版本
  -> 后继 kernel 再次复制完整输入
```

即使 kernel 本身获得接近线性的加速，每一级 `O(V)` 的复制与合并仍可能超过计算收益。
要接近 stencil 或长计算 pipeline 的理想表现，关键是让 `part_i` 产生的数据直接成为
下一 kernel 在同一设备上的 `part_i` 输入，并只在真正不兼容的边界物化。

但 runtime 仅看到 accessor mode、range 和 NDRange，不能据此证明 kernel 内部没有
间接索引、halo/global read 或转置访问。因此，跳过数据移动必须建立在显式访问合同
和版本证明上，不能依赖乐观猜测。

---

## 2. 系统模型与不可破坏的不变量

### 2.1 wait-delimited DAG

handler 拦截 kernel command group，在用户触发 `queue::wait()` 或相应同步点前暂缓实际
执行。一个 wait 之前已经提交的 kernel 形成一个调度窗口，handler 向 daemon 发送：

- 稳定 kernel identity；
- NDRange 维度与 global size；
- accessor 的读写模式、元素宽度、buffer size；
- 实际 access range、offset、sub-buffer 属性；
- 可选的 partition-local contract。

daemon 根据 buffer/accessor 冲突建立 RAW、WAR、WAW 依赖，并将窗口表示为 DAG。
历史节点仍可作为数据 producer 和 residency anchor，但已完成窗口的合成 HEFT 时间不能
继续污染单 rank 下一窗口的可用时间。

### 2.2 handler、daemon 和 queue 的职责边界

```text
用户提交
  -> handler 捕获 command group 与 requirement
  -> wait 暴露一个完整窗口
  -> daemon 建 DAG、预测、选择候选
  -> handler 按 dispatch wave 重提交
  -> event/Split parts 完成
  -> handler 报告 completion 和可选 profile
  -> daemon 释放资源、更新模型、重新选择
  -> 窗口全部完成后用户 wait 返回
```

- daemon 是 placement、Split factor、device subset 和 dispatch admission 的唯一决策者；
- handler 负责验证实际 command group、准备数据、创建/复用设备 queue、启动 kernel、
  维护分区版本以及报告完成；
- handler 可以因正确性把不安全的 resident Split 降级为 materializing Split，但不能
  另行选择一个“更快设备”覆盖 daemon；
- queue 只执行已批准的 dispatch wave，不再叠加一套 monitor penalty 或 placement
  heuristic。

### 2.3 五个不变量

1. **尊重用户 fence**：runtime 可以调度所有已经提交且依赖满足的 kernel，但不能越过
   用户 `wait()` 推测下一 epoch 尚未提交的工作。
2. **完成与 profile 解耦**：`duration_ns=0` 仍是合法 completion，必须释放设备；缺失
   profile 不能阻止 READY kernel。
3. **统一候选入口**：Single、Split、迁移、monitor、数据局部性和不确定度必须汇入同一
   selector，不能有组件在事后独立改卡。
4. **Split 以 gang 管理**：一个 Split kernel 同时占用所有 part devices，只有全部 part
   完成以及该 mode 所需的 materialization 完成后，才具有相应完成语义。
5. **数据有效性需要证明**：设备上存在 allocation 不等于该设备拥有当前 logical
   version；只有 region/version 合同满足时才能复用，否则 materialize 或回退 Single。

---

## 3. 核心贡献一：不确定性感知的分层性能预测与统一风险决策

### 3.1 动机

原系统中 cold estimate、EWMA profile、monitor penalty、通信 cost、Split threshold 和
migration rule 分别起作用。同一个事实可能被重复计费，例如 monitor 一方面增加合成
available time，另一方面再乘执行 penalty；某种 Split 可能通过独立 hysteresis 被否决，
即使它在 HEFT 目标中更优。这使最终 placement 无法解释，也使消融实验难以定义。

本贡献的目标是把每一种信息转换成同一种预测对象，并只通过一个风险目标影响调度：

```text
CostEstimate {
  mean,
  uncertainty,
  samples,
  source
}
```

这里 `uncertainty` 表示下一次 service time 的预测标准差尺度，而不是样本均值的标准误。
运行抖动不会因为样本变多而被错误地消除；模型未知性则可以随 exact 样本增加而衰减。

### 3.2 稳定 profile identity 与 mode 隔离

handler 使用稳定 FNV-1a kernel symbol hash，而不是跨进程不稳定的 `std::hash`。完整
profile key 包含：

```text
kernel_identity
work_dim + global_size
requirement count
每个 accessor 的 mode、elem_size、buffer size
memory range、access range、offset、sub-buffer
partition_local 标志
总 read/write bytes
```

profile table 的执行 mode 还包含：

```text
(rank, device, num_parts, persistent_split)
```

因此以下样本不会混合：

- 相同 shape 但 kernel symbol 不同；
- Single 与 2-way/4-way Split；
- 每次都 canonical merge 的 SplitMaterialize；
- writable partition 留在设备上的 SplitResident。

这很重要，因为四种 mode 的计时边界和数据维护成本不同，把它们放进一个 EWMA 会让
调度器学习到没有物理含义的平均值。

### 3.3 exact profile：均值、波动与新近性

每个 exact entry 保存：

- EWMA service cost；
- Welford online mean、`M2` 和标准差；
- min cost 和样本数；
- 本次 daemon 进程内的 live sample 数；
- 最近 observation 的 wall-clock 时间；
- 采样设备的 FP32/FP64 capability。

对 exact profile，预测不确定度由两部分组成：

```text
observed_jitter = sample standard deviation
prior           = ewma * prior_error / sqrt(samples)
uncertainty     = hypot(observed_jitter, prior)
```

第一项表示不可通过增加样本消除的运行时波动，第二项表示随观测增加而衰减的认知未知性。

### 3.4 持久化 profile：让历史帮助冷启动但不冒充 live exact

rank 0 daemon 启动时加载 append-only observation journal。每个 kernel 完成后：

```text
在 profile lock 内更新内存 EWMA/Welford
  -> 释放 profile lock
  -> 向有界 store queue enqueue observation
  -> completion loop 可以立刻继续

后台 writer thread
  -> append record
  -> flush
```

存储队列满时丢弃最旧的待落盘 observation，而不是阻塞 completion；当前进程的内存
profile 已经更新。daemon 退出时才 drain 并 join writer。这样持久化不会重新引入“为了
profile 等待”的控制面 barrier。

V2 record 保存 namespace、profile key、mode、duration、时间、源设备 capability 和
kernel feature。`SYCL_SNMD_PROFILE_NAMESPACE` 用于隔离应用/构建，正式运行应设置为
应用版本、git commit 或二进制哈希。默认 `default` 兼容旧记录，但无法自动识别同名
kernel 的实现已经改变。

loaded exact 具有额外 stale floor：基础为 25%，并按每天 1% 增加，额外年龄项最多
50%。因此 persisted sample 可以减少完全盲目的冷启动，但其风险不会被当成本次运行
的新鲜实测。只有 live exact 才能作为“当前设备配置曾成功分配”的 memory-fit 旁证；
历史记录不能绕过当前显存检查。

### 3.5 分层迁移：从 exact 到 learned cold start

Single 和 Split estimator 使用同一命中顺序：

```text
live exact
  -> persisted exact
  -> 同 exact key、同 mode 的跨设备 capability scaling
  -> 同 kernel identity 的跨 shape learned estimate
  -> 严格 structural cohort 的 learned estimate
  -> analytical cold model
```

跨设备 scaling 使用记录 observation 时的 capability，而不是本次进程中同一个
`rank/device` 编号的当前能力。scaled profile 传播源样本波动，并额外加入 20% target
capability model error，不能冒充 exact。

同 identity 的 shape transfer 使用以下与 DAG 宽度无关的工作量代理：

```text
shape_work = max(global_items, total_access_elements)
work_scale = clamp(target_work / sample_work, 0.05, 20)
```

这是尺度代理，不试图仅凭字节数推断算术强度；同一 kernel 的实测 service time 已包含
其计算密度。same-identity learned estimate 具有至少 20% 的模型误差，并随 feature
distance 增大。

跨 kernel 迁移更加保守。只有以下结构完全一致时，样本才属于同一 cohort：

- work dimension；
- requirement、read requirement、write requirement 数；
- dominant element size；
- access-mode mask；
- partition-local read/write 属性。

同时 shape distance 必须很小，模型误差从 60% 起步。一旦存在同 identity neighbor，
预测器不再混入 structural neighbor。多个 neighbor 的加权二阶矩同时保留各自的不确定度
和 neighbor disagreement，避免平均值看似稳定却掩盖样本冲突。

### 3.6 analytical cold 与 derived Split

完全没有可用 profile 时，cold model 根据实际 access range、precision、NDRange、读写
结构、启发式 arithmetic intensity 和设备 capability 计算 service mean，并给出 50%
模型不确定度。

没有 Split profile 时，derived Split 从最佳 Single estimate 推导：

```text
split_mean = best_single_mean / (parts * 0.85)
           + predicted_input_replication
           + predicted_output_merge
           + launch_overhead
```

它传播 Single uncertainty，并加入 30% Split model error。cold Split 仍有 bounded probe
可行域：Single 预测必须足够长，且预测收益至少达到门槛；宽 DAG 已能占满 GPU 时，
未经 profile 证明吞吐收益的 Split 会被拒绝。这些规则限制未知 mode 的探索范围，不是
profiled candidate 的第二套长期目标。

### 3.7 monitor 与 profile 的统一方式

每次 scheduling pass 开始时 daemon 对 monitor、device capability 和 free memory 取一次
一致快照。monitor 只产生一个 `external_load_service_scale`：

```text
util <= threshold       -> scale = 1
fresh program profile   -> scale = 1
otherwise               -> scale = 1 + external_busy_ratio
```

刚在该设备完成并上报的本程序 profile 会抑制短时间内陈旧的 NVML busy 值，避免把同一
工作既作为已完成 profile，又作为未来 external load 重复惩罚。monitor scale 同时乘
`mean` 和 `uncertainty`，不再额外伪造 available-time delay。

### 3.8 通信不是独立 penalty，而是资源化的随机代价

通信模型根据 producer placement、consumer candidate、有效字节数和 H2D/D2H/D2D/
same-node/cross-rank bandwidth 估计传输。每个 transfer 同样产生 mean 和 15%
uncertainty。

更重要的是，dependent transfer 不只加在 predecessor finish 上。daemon 计算：

```text
transfer_start = max(predecessor_complete,
                     source_calendar,
                     destination_calendar)
```

并同时预留 source 与 destination endpoint。这样，一个数据副本若必须从正在执行长
kernel 的 source queue 发出，调度器不会把它当作可以立即完成的常数 copy。

### 3.9 唯一风险目标与候选选择

每个 READY node 枚举：

```text
Single(rank, device)
Split(rank, ordered device subset, num_parts, materialize/resident)
```

候选统一计算：

```text
start = max(target compute calendar, dependency transfer ready)
mean_finish = start + execution.mean

total_uncertainty = hypot(execution.uncertainty,
                          transfer.uncertainty)

risk = mean_finish + beta * total_uncertainty
beta = 1.96
```

`selectUnifiedTaskCandidate()` 是 batch HEFT 与 completion dispatcher 的共同入口。风险
相同时才依次偏好：更少数据移动、更少 part、更早 mean finish、更小 rank/device 编号。
这种确定性 tie-break 既保留数据局部性，也避免在风险等价时无意义地占用 gang。

这一贡献的关键不是某个预测公式本身，而是：**所有计算、通信、monitor 和 mode 信息
都被归一为可传播的不确定度，并且只有一个决策目标有权改变 placement。**

---

## 4. 核心贡献二：完成状态驱动的负反馈 HEFT 与 ready queue

### 4.1 动机

静态 HEFT 的 upward rank 对 DAG criticality 很有价值，但静态 device calendar 只能反映
预测。把所有 kernel 一次性发送给 handler，会把预测 placement 固化到设备 queue 中：
实际较快的 GPU 即使提前完成，也可能没有新的可运行 kernel；实际较慢的 GPU 后面却
排着多个任务。

本设计保留 HEFT 的全局 DAG priority，但把资源释放与 placement 变成 completion-driven：

- rank 负责回答“哪个 READY node 更重要”；
- completion 回答“哪些资源现在真的空闲”；
- 统一 selector 在每次资源变化后重新回答“这个 node 现在应放在哪里、是否 Split”。

这形成针对 prediction error 和 execution jitter 的负反馈。

### 4.2 状态机：完成状态与性能观测分离

概念状态机为：

```text
PENDING --predecessors complete--> READY
READY   --candidate admitted-----> DISPATCHED
DISPATCHED --actual events done--> COMPLETE
COMPLETE --duration available----> PROFILED
```

实现中 `READY` 由 predecessor phase 动态计算，持久状态为
`Pending/Dispatched/Complete`。`PROFILED` 不是资源状态：completion message 中
`duration_ns=0` 仍然使 node 进入 Complete 并释放 reservation，只有合法 duration 才
更新 cost table。

这解决了此前“为了得到 profile 再等一次 event”或“为了让所有 rank 的 profile 到齐
再调度”的问题。profile 更新影响未来预测，但不能成为当前 READY admission 的前置条件。

### 4.3 daemon 的 rolling dispatch

单 daemon/rank completion window 的核心循环是：

```text
计算完整 DAG 的 HEFT upward rank
初始化每个 node 的 runtime phase

while not all COMPLETE:
    找到所有 predecessor COMPLETE 的 PENDING nodes
    按 upward rank 选择高优先级 READY node
    调用 selectUnifiedTaskCandidate(node)
    若候选设备当前可用：
        原子预留 Single device 或 Split gang
        加入本 dispatch wave
    发送 DISPATCH_BATCH_V1

    等待至少一个 completion batch
    校验 wait_count、kernel_count、mode 与重复 completion
    释放真实完成 kernel 的 compute reservations
    更新 producer placement / persistent mode
    duration 有效时更新 profile
```

被占用的 compute device 在动态 calendar 中表示为 infinity，因此不会预排第二个 compute
kernel。只有实际 completion 到达后，该设备才重新进入候选集合。Split 以 gang 方式
同时预留全部 part devices，避免两个 logical kernel 共享同一 part device。

### 4.4 transfer reservation 与 compute reservation 的不同生命周期

compute reservation 持续到真实 kernel completion；transfer reservation 只用于同一个
dispatch wave 内的冲突排序。wave 发出后，稳定 in-order queue 保证实际 copy/compute
顺序，daemon 不把一次毫秒级 D2D copy 的 source GPU 锁到目标数十秒 kernel 完成。

这一差异解决了两个相反错误：

- 不预留 transfer endpoint会高估 copy 并行度；
- 把 source endpoint占用到 consumer kernel 完成又会制造数十秒虚假 busy。

### 4.5 handler 的 completion-driven queue 支撑

handler 不再一次性提交完整窗口，而只提交当前 daemon 批准的 dispatch wave：

1. 根据 daemon 决定设置 actual device、part 数、ordered split devices 和
   `persistent_split`；
2. 对即将运行的 kernel，先处理其触及的旧 Split version：兼容则保留，不兼容则物化；
3. 复用每设备 private-context、in-order、profiling queue 并调用 `resubmit()`；
4. 将 event 和计时边界登记为 in-flight；
5. 约每 1 ms 查询 event 状态；
6. 对已完成 single event 调用 `wait()` 传播异步错误并读取 profiling timestamp；
7. Split 等待全部 part，按 mode 完成必要物化；
8. 将同时完成的 kernel 合并为一个 `COMPLETION_BATCH_V1`；
9. 等待 daemon 的下一个 dispatch wave。

为了避免 handler 的顺序 `resubmit()` 先被一次迁移阻塞，daemon 在一个 wave 内优先发送
零移动 kernel，再发送需要移动数据的 kernel。这只改变 host submission 顺序，不改变
统一 selector 已决定的 placement。

### 4.6 wait 语义、profile 传播与错误处理

用户 `wait()` 仍是最终 fence。runtime 只调度该 wait 前已经提交的节点；如果应用在
每个 epoch 结束后 wait，下一 epoch 尚未提交，runtime 不可能让早完成 scenario 越过
这个 fence。

普通 profile 路径中，handler 的 fence wait 与 timestamp 查询已经分离；timestamp 查询
不再次调用 event wait。跨 rank profile 传播使用 worker `MPI_Isend` 与 master
`MPI_Iprobe` drain，不要求所有 rank 在每个 profile batch 同步；只有退出时 flush。

动态 completion 协议一旦发送第一批 dispatch，就禁止窗口中途静默降级为静态协议：

- 未知、重复或非 Dispatched completion 被视为协议错误；
- ready queue 无 READY 且无 in-flight，但窗口未完成时视为 deadlock；
- handler submission、event wait 或 materialization 抛错时发送 failure completion；
- daemon 发送 `window_failed`，用户 wait 获得 runtime error；
- 只有首个动态 dispatch 尚未发出前，才允许完整恢复保存的 placement/calendar 并使用
  batch-static fallback。

这保证 fallback 不会在部分 kernel 已执行后形成“双重执行”或让 wait 错误提前返回。

### 4.7 调度状态的隔离范围

每个应用 daemon thread 使用 thread-local capability、monitor scale、memory snapshot 和
HEFT/completion calendar，避免一个应用覆盖另一个应用的 DAG 时间轴，也不需要持有跨
数十秒 kernel 的全局锁。共享 profile 和 monitor 原始数据通过短锁与快照访问。

当前这种设计解决的是**单个应用调度上下文内部的全局决策**。多个应用争用同一 GPU
时，影响只能在下一次 monitor snapshot 中作为 external load 反馈；还没有一个跨应用
的全局 device lease arbiter。这是当前“global scheduling”声明必须限定的边界。

---

## 5. 核心贡献三：contract-proven partition-resident Split dataflow

### 5.1 动机

Split 是否值得不能只看 `single_time / parts`。真正的 mode 应包括计算与数据语义：

```text
Single
SplitMaterialize = prepare + gang compute + canonical merge
SplitResident    = partition prepare/reuse + gang compute
                  + future incompatible-edge materialization if needed
```

如果每个 Split kernel 都立即 merge，连续四级 pointwise/stencil pipeline 会支付四次全量
数据维护；若分区可以跨级留在同一组 GPU，前三次中间 merge 可以消失。因此“持久化
Split”既是数据层机制，也是调度候选的一种独立 mode。

### 5.2 普通 SplitMaterialize 的安全基线

未声明 partition contract 的应用保持普通 Split：

- 当前主要沿 NDRange dim0 划分；
- part 数目前为 2 或 4；
- global dim0 与 writable accessor dim0 必须可整除；
- writable region 必须能表示为连续 row block；
- offset、sub-buffer、非连续 layout 或不匹配的二维列访问会拒绝 Split；
- 每张 part GPU 准备所需输入并执行自己的 subrange；
- 输出最终 canonicalize 到 ordered split device 集合的首卡；
- direct D2D 失败时回退 D2H→H2D。

同一 wait window 内，普通 replicated read-only input 有 replica cache；后续 Split 在相同
queue/context 可复用，任何写访问在提交前使 cache 失效。该路径以额外数据移动换取
保守正确性，也是 resident mode 不满足合同时的 fallback。

### 5.3 显式、逐 accessor 的 partition-local contract

runtime 新增实验接口：

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

合同语义是：每个 Split part 只访问与其 dim0 work-item block 对应的 accessor row block。
声明按 accessor 而不是按 kernel：

- pointwise input/output 可以标记；
- replicated coefficient table 不标记；
- halo/global read、间接索引、全局归约和 atomic 不得标记。

handler 捕获声明时先检查 shape，再把 `partition_local` 写入 protocol metadata。daemon
用 metadata 做候选与通信预测；handler 在执行真实 command group 前再次验证。即使
daemon 因 stale metadata 或实现错误给出过度乐观 resident 决定，handler 也只会把它
降级为 materializing Split，不会削弱内存正确性。

### 5.4 resident admission

只有同时满足以下条件，daemon 才构造 SplitResident candidate：

- 单 daemon/rank；
- 所有 writable accessor 都有 partition-local contract；
- 不包含 atomic；
- accessor 非 sub-buffer、offset 为零；
- access range 覆盖合法 dim0/trailing layout；
- global range 与 writable range 可按 part 数整除；
- DAG 中存在可以使用相同 partition scheme 的 partition-local successor；
- producer/consumer 的 parts、ordered device set 与 partition boundaries 一致。

“存在 compatible successor”很重要：若 Split 后立刻面对不兼容 consumer 或窗口结束，
resident mode 没有可兑现的数据复用，普通 materialization 语义更直接。

### 5.5 window-local version ledger

handler 的 `PendingOfflineSplitMerge` 不再只是延迟 merge 列表，而承担 window-local
partition version ledger，记录：

```text
producer kernel
part events / completion time
resident flag
ordered split device set
partition dim0
per-part queue 与 requirement copy
read buffers / partition-local read buffers
written buffers
superseded written buffers
```

producer SplitResident 完成时，handler 只等待所有 part 并保留 per-device allocation，
不执行 canonical merge。compatible successor 在相同 stable in-order queues 上直接消费
对应 part。首次 partition-local input 也只向每卡移动 own block，而不是给每卡完整
snapshot。

当新 producer 写同一 buffer 时，旧 resident version 被标为 superseded。最终物化时
跳过 superseded part，防止 stale kernel output 覆盖较新版本。相同 scheme 上的 RAW、
WAR 和 WAW 通过 stable in-order queues 保序；若 kernel 以不兼容方式触及 pending write
或破坏 partition-local read 的反依赖，handler 先物化旧版本。

### 5.6 compatible edge、incompatible edge 与物化边界

数据路径分为：

```text
compatible resident producer -> resident consumer
    对应 part 已在对应设备：edge transfer = 0

resident producer -> incompatible Split/Single consumer
    gather dirty parts 到 producer ordered device set 首卡
    再按 consumer mode 准备完整或分区输入

resident producer -> 用户窗口 fence
    materialize canonical version
```

daemon 与 handler 使用同一个 canonical source 约定。daemon 在 incompatible edge 的
通信模型中估计 gather 与后续 replication，并在 dispatch wave 中预留隐藏的 source/
destination endpoints；handler 按相同顺序实际执行 D2D 或 host-staged fallback。

这使“调度器认为 compatible edge 为零通信”和“执行器确实复用该 partition”保持一致，
避免模型与数据层各自维护不同真相。

### 5.7 Split profile 的计时语义

普通 Split profile 记录：

```text
prepare/host wall + all part completion wall + materialization duration
```

resident Split profile 记录：

```text
prepare/host wall + all part completion wall
```

未来的不兼容 materialization 作为 dependency transfer 计入相应边，而不再次折入
producer resident profile。`persistent_split` 作为 profile mode key 的一部分，防止两种
计时语义混合，也防止静态与 completion 路径对同一 mode 学到不同含义。

### 5.8 当前 full virtual allocation 的取舍

当前每张 part GPU 仍保留完整虚拟 buffer allocation，kernel 可以继续用 global id
索引；只是每张卡只有 own block 拥有当前有效版本。优点是不改 kernel/accessor ABI，
可以先消除中间级全量复制和 merge；代价是显存占用仍按完整 buffer 计算。

因此 daemon 的 memory admission 必须按每张卡完整 allocation 检查，不能把 writable
storage 错算为 `bytes / parts`。真正降低显存需要 accessor base/offset remap 或
distributed buffer ABI，是下一阶段而不是当前 resident correctness 的一部分。

---

## 6. 三项贡献如何组合成一个完整故事

### 6.1 统一问题：在不确定条件下选择并维持正确的并行形态

三项贡献不是“模型优化 + queue 优化 + copy 优化”的并列集合。它们分别回答同一个
闭环中的三个问题：

| 闭环阶段 | 核心问题 | 对应贡献 |
|---|---|---|
| 预测 | Single、迁移或某种 Split 的完成时间和风险是多少 | 分层不确定性感知模型 |
| 调度 | 当前真实空闲资源下，哪个 READY node 应以哪种 mode 执行 | completion-driven HEFT |
| 数据状态 | 预测的数据局部性和零通信边在执行时是否真实成立 | contract-proven resident Split |
| 反馈 | 实际 service time 与 mode 结果如何影响下一次决策 | completion/profile feedback + 持久化模型 |

任何一项单独存在都不完整：

- 只有预测模型、没有 completion queue：一次预测误差仍会固化到整个静态 batch；
- 只有 completion queue、没有统一预测：GPU 虽能被及时释放，下一次仍可能被互相冲突
  的 monitor/profile/Split heuristic 错误选择；
- 只有 Split、不维护 partition version：kernel 加速可能被每级复制与 merge 完全抵消；
- 只有 resident data、不让调度器感知 mode：调度器可能改变 device subset，立即破坏
  residency，或在宽 DAG 中错误让 Split 抢占 task parallelism；
- 只有在线 profile、不持久化且不表达风险：每次运行仍经历同样的冷启动错误，也无法
  区分可靠 exact 与弱迁移样本。

### 6.2 完整运行链路

```text
1. 用户在 wait 前提交 kernel DAG
        |
2. handler 捕获 kernel/accessor/partition contract
        |
3. daemon 构造依赖、HEFT priority 与设备快照
        |
4. 分层模型为 Single / SplitMaterialize / SplitResident
   生成 mean + uncertainty
        |
5. 通信模型结合 producer region/device state，生成 transfer plan
        |
6. 统一 selector 以 risk-EFT 选择 READY node 与 device subset
        |
7. handler 按 dispatch wave 准备数据并提交到稳定 queue
        |
8. actual completion 释放 Single device 或 Split gang
        |
9. resident output 更新 partition version；不兼容 output 被 materialize
        |
10. duration 更新 live profile，并异步写入持久化 store
        |
11. daemon 在新的真实资源状态上重新选择下一批 READY node
```

这个链路同时形成两种负反馈：

- **资源负反馈**：慢设备没有完成就不会释放；快设备一完成立即重新参与候选，消除
  静态波次的长尾空洞；
- **模型负反馈**：实际 duration 更新对应 device/mode/shape 的分布，后续 placement
  和下次 daemon 冷启动都能修正。

partition version 则把反馈从“计算用了多久”扩展为“数据现在在哪里、哪一部分有效”。

### 6.3 task parallelism 与 Split 的统一解释

该故事最有价值的点不是“总能 Split”，而是 runtime 可以拒绝不合适的 Split。

以四张 A6000 为例：

- 四个独立 radio scenario、每个数秒：DAG width 已等于 GPU 数，whole-kernel placement
  能接近逻辑上限；wide-DAG guard 应拒绝未经 profile 证明吞吐收益的 Split；
- 一个数十秒 scenario：task parallelism 不足，若 derived/learned Split risk 加数据
  移动仍小于 Single，调度器可选择 2-way/4-way gang；
- 一个四级 partition-local chain：若每级都能复用相同 device subset，SplitResident
  消除中间 merge，Split 代价会随 profile 收敛；
- 一个带间接 gather 或未知 halo 的 kernel：即使计算很大，也不能谎报 resident；应走
  materializing Split 或 Single。

因此，Split 数量不是成功指标。正确指标是：runtime 是否在 DAG 宽度、预测风险和数据
生命周期允许时选择正确的并行形态，并在预测误差出现后及时恢复设备利用率。

### 6.4 与数据流 runtime 的互补定位

类似 Celerity 的 range mapper/virtual buffer 机制在 region correctness、halo 和部分物理
allocation 上更成熟；其 mapper 主要回答“一个 execution chunk 需要哪些数据”。本系统
的核心差异应表述为：

- 将 compute cost、设备异构/负载和历史 profile 与 data requirement 分离建模；
- 对 whole-kernel placement、参与设备集合和 Split factor 做风险感知联合选择；
- 用实际 completion 而非仅静态 command graph 推进资源状态；
- 仅在显式合同证明时利用 partition residency，不能把当前实现描述成已经具备通用
  virtual buffer 或 neighborhood mapper。

完整故事不是否定 range-based dataflow，而是说明：**访问区域语义解决 correctness，
成本与完成反馈解决是否值得以及何时执行；两者需要在候选选择中融合。** 当前
partition-local contract 是两者结合的第一个受限实现，SplitHalo 是尚未完成的扩展。

### 6.5 建议的三条论文贡献表述

在不扩大当前实现范围的前提下，可表述为：

1. **一种可持久化的分层不确定性感知 kernel/mode 性能模型**：统一 exact、stale、
   capability-scaled、shape-transferred 和 analytical cold estimate，并将计算与通信
   uncertainty 融入同一 risk-EFT objective。
2. **一种 completion-driven feedback HEFT runtime**：保留 DAG criticality priority，
   但以真实 completion 管理 device/gang reservation，并在每次资源变化后联合重选
   whole-kernel placement 与 Split mode。
3. **一种 contract-proven partition-resident Split dataflow**：显式逐 accessor 声明
   partition locality，在兼容 producer-consumer chain 中维持 writable partition version，
   对不兼容边安全 canonicalize，并让其通信与计时语义进入统一调度器。

总标题或核心命题可以是：

> **Uncertainty-Aware Feedback Scheduling for Adaptive Task and Persistent
> Partition Parallelism in SYCL DAGs**

---

## 7. 与原先目标贡献的完成度对照

原先设定的三项目标是：

1. 统一的不确定感知预测模型，包括更精确的冷启动；
2. 有负反馈的调度算法，包括 kernel 完成状态驱动的 queue，profile 真正进入调度器；
3. 持久化数据的 Split，使 Split 不再被重复全量数据移动压垮。

仅从功能实现而非实验效果判断：

| 原目标 | 当前已实现 | 尚未实现 | 功能判断 |
|---|---|---|---|
| 统一预测模型 | 统一 `CostEstimate`、Welford/EWMA、来源区分、UCB、monitor/communication uncertainty、持久化与层次迁移 | compiler-level kernel features、自动 binary fingerprint、系统化 exploration/calibration | runtime 预测—决策接口已闭环；“冷启动精度充分”尚不能声称 |
| 负反馈调度 | 单 rank completion-driven ready queue、真实 device/gang release、统一重选、非阻塞 profile、失败协议 | multi-rank completion/data-ready、跨应用全局 lease、远端失败恢复 | 单节点单应用范围闭环；多节点/多应用全局调度未闭环 |
| 持久化 Split | window 内 strict partition-local chain residency、version supersede、兼容零边通信、不兼容 materialize、mode profile 隔离 | halo/region mapper、跨 wait residency、物理分区 allocation、跨 rank partition | pointwise/own-block chain 闭环；通用 stencil/分布式 buffer 未闭环 |

---

## 8. 尚未实现的设计与原因

### 8.1 贡献一剩余：真正更强的 cold-start 模型

当前 learned cold start 比纯 analytical cold 更有信息，但跨 kernel 的 structural cohort
仍只能看到 NDRange 与 accessor 元数据。它不知道：

- 指令混合与实际 FLOP 数；
- register pressure、shared/local memory；
- occupancy、work-group shape 与 launch constraints；
- cache reuse、coalescing 和 divergent branches；
- data-dependent inner loop、稀疏行长度或 Monte Carlo 实际 collision 分布。

因此尚缺 compiler/runtime integration，例如在 kernel metadata 中携带静态 instruction
features，或由可选用户 cost class 描述隐藏计算强度。daemon 应消费这些特征并学习
residual，而不是继续从 buffer bytes 增加经验倍数。

还缺以下机制：

1. **自动 build identity**：当前 namespace 由 daemon 环境变量指定，不能自动区分同名
   kernel 的二进制变更，也不适合一个 daemon 同时服务不同 namespace 的多个应用；
2. **受控反事实探索**：调度器只学习实际选择的 mode；若 cold guard 永远不选择某个
   Single/Split/device，就没有反事实 profile。当前 bounded Split probe 不是完整 bandit；
3. **concept drift/outlier**：EWMA 能跟踪变化，但没有显式 change-point、robust outlier
   filter 或 per-phase model；
4. **profile store 生命周期**：journal 是 append-only 且只有 TTL 过滤，没有 compaction、
   schema migration 管理和容量上限；
5. **校准闭环**：尚未按 estimate source 验证 coverage、误差分位数与 95% 风险区间是否
   校准。该项主要属于实验，但校准结果可能要求调整模型结构。

### 8.2 贡献二剩余：multi-rank completion-driven scheduling

当前 multi-rank 仍使用 batch-static fallback。正确实现不能只增加一条 remote
`COMPLETE`，而应定义 master-owned lease protocol：

```text
DISPATCH(window_id, node_id, lease_id, rank, devices, input_versions)
EXEC_ACCEPTED(window_id, node_id, lease_id)
COMPUTE_COMPLETE(window_id, node_id, lease_id, output_versions)
DATA_READY(window_id, node_id, lease_id, regions)
```

尚缺的关键语义：

- `(window,node,lease)` 幂等，迟到 ACK 不能释放新 reservation；
- compute completion 与跨 rank data-ready 分离；
- master 同时维护 remote device lease 与 distributed buffer version；
- timeout 不能盲目重跑可能有副作用的 kernel；
- 动态协议启用前做 capability negotiation，启用后不得中途静默 fallback；
- daemon/rank failure、MPI progress 和 late message 的 fail-stop/recovery 规则。

此外，当前每应用 thread-local calendar 不是跨应用 device arbiter。若论文目标包含多租户
全局调度，还需要 daemon 级共享物理 device lease、优先级/公平性和 admission control；
monitor feedback 只能事后看到竞争，不能替代协调。

### 8.3 贡献二剩余：更真实的 transfer completion

当前 transfer endpoint reservation 改善了候选预测，但 completion 协议主要报告 logical
kernel completion；同 wave 的 transfer calendar 是预测性短期 reservation，实际顺序依靠
in-order queue。若未来要支持 out-of-order queue、独立 copy engine 或跨 rank overlap，
需要把 copy/data-ready event 提升为显式调度事件，否则 daemon 无法知道 copy engine
何时真实释放。

handler 的 `resubmit()` 也可能因跨 context 数据准备在 host 侧同步阻塞。当前通过 wave
ordering 减轻“先阻塞后饿死其他 GPU”，但没有独立异步 submission engine。若 trace
证明首波 kernel start 仍有明显 skew，应把数据准备拆成 per-device asynchronous command
并让 completion/data-ready 协议感知，而不是继续调整 HEFT penalty。

### 8.4 贡献三剩余：SplitHalo 和通用 region version

当前 `partition_local` 是 own-block 布尔合同，不能表达 stencil neighborhood。需要新的
mapper descriptor：

```text
PartitionAccess {
  kind = local | halo | replicated | reduction,
  split_dim,
  left_halo,
  right_halo,
  boundary_policy
}
```

以及通用 region ledger：

```text
RegionVersion {
  buffer_id, logical_version, partition_scheme,
  owned_region[part], owned_version[part],
  ghost_region[part], ghost_source_version[part],
  canonical_regions
}
```

第一阶段可保留每卡完整 virtual allocation，只传相邻 halo；这已经能把连续 stencil 的
每级通信从全量 `O(V)` 降到边界量。若要 overlap interior compute 与 halo copy，还需把
一个 logical part 拆成 interior/boundary launches，并在 profile/completion 中仍作为一个
logical kernel mode 汇总。

### 8.5 贡献三剩余：物理分区 allocation

当前每卡完整 allocation 无法支持“数据集大于单 GPU 显存”或真正 `bytes/parts` 的显存
收益。下一阶段需要：

- accessor base/offset remap；
- global id 到 local allocation 的地址变换；
- owned+halo allocation growth；
- sub-buffer/alias 与 region copy 的一致语义；
- kernel ABI/compiler support。

这是数据布局贡献，不能通过修改 daemon memory estimate 假装已经实现。

### 8.6 贡献三剩余：跨 wait、跨 rank 与动态 repartition

当前 resident version 只在同一 wait window 内保留，窗口结束前 canonical materialize。
跨 wait residency 要把 version ledger 从 handler pending vector 提升到 buffer 生命周期，
并在 host accessor、buffer destructor、interop handle、未知 queue/context consumer 时按需
materialize。`queue::wait()` 本身只要求执行完成，不天然意味着 host 读取，但在这些生命周期
hook 完成前，窗口末 materialization 是安全边界。

跨 rank partition 还要求 region-aware send/receive、remote version owner 和 data-ready
ACK；动态 repartition 则需要比较旧 scheme→新 scheme 的 shuffle cost。它们应建立在
multi-rank lease 与 region ledger 之上，不能单独增加一个 `num_parts`。

当前也未支持：

- 非 dim0/tile/column partition；
- reduction 的局部结果与最终 combine；
- overlapping writable regions；
- atomic/间接 scatter 的安全 distributed semantics；
- 一般 sub-buffer 和非零 offset 的 persistent Split。

---

## 9. 推荐的后续演进路线

### 9.1 若论文主范围保持单节点多 GPU

优先顺序应为：

1. 验证当前 completion/profile/store/ResidentSplit 的正确性与协议一致性；
2. 做 estimate source calibration，确认 persisted/learned uncertainty 是否覆盖真实误差；
3. 增加 compiler kernel features，提高跨 kernel cold start；
4. 实现 `SplitHaloFullAllocation`，让 stencil 在不改 ABI 的情况下只交换 halo；
5. 再根据显存需求决定是否实现 physical region allocation。

这条路线与当前完整故事最一致，也不要求在论文中宣称尚未闭环的多节点动态调度。

### 9.2 若论文必须以多节点为核心

应把下一阶段优先级改为：

1. dispatch lease 与幂等 remote ACK；
2. compute completion / data-ready 分离；
3. distributed region version；
4. timeout、late ACK 和 rank failure；
5. 最后才允许跨 rank Split 与 dynamic repartition。

在这些机制完成前，多节点应诚实使用 static fallback，不能把非阻塞 profile 传播等同于
multi-rank completion-driven scheduling。

### 9.3 论文叙事上的取舍

最合适的当前主线是“自适应选择 task 与 persistent partition parallelism”，而不是：

- “所有大 kernel 都应 Split”；
- “已经实现通用 distributed SYCL buffer”；
- “monitor-aware HEFT 的若干 heuristic”；
- “通过 daemon 给原生 SYCL 套一个 HEFT”。

论文应围绕一个反事实问题组织实验：

> 在相同 wait-window DAG 上，如果没有 uncertainty、没有 completion feedback、没有
> partition residency，runtime 会分别作出什么错误决定；三者组合后如何接近该 DAG
> 在当前设备集合上的可实现下界？

这样三项贡献具有明确的因果关系：预测决定并行形态，completion 修正资源时间，resident
dataflow 兑现局部性，profile 再把结果反馈给下一轮预测。

---

## 10. 实现位置索引

主要实现文件如下：

- `code-llvm-sycl/sycl/source/detail/daemon/daemon.cpp`
  - `CostEstimate`、profile statistics、persistent store、learned estimate；
  - monitor/device snapshot、通信 profile 和 endpoint reservation；
  - `selectUnifiedTaskCandidate()`；
  - batch HEFT 与 single-rank completion-driven loop；
  - Split admission、resident edge communication model。
- `code-llvm-sycl/sycl/source/detail/daemon/daemon.hpp`
  - kernel/accessor/profile/scheduling/completion protocol；
  - stable kernel identity 与 profile key；
  - `persistent_split`、`partition_local` 字段。
- `code-llvm-sycl/sycl/source/handler.cpp`
  - command group metadata 捕获与 profile key；
  - per-device queue、Split prepare/launch/merge；
  - completion polling/batch acknowledgement；
  - `PendingOfflineSplitMerge` version ledger、replica cache、supersede 与 fallback。
- `code-llvm-sycl/sycl/include/sycl/handler.hpp`
  - 实验接口 `ext_snmd_partition_local(accessor)`。
- `code-llvm-sycl/sycl/source/detail/daemon/define.hpp`
  - completion queue、Split、cold probe、wide-DAG guard、risk/uncertainty 和 profile-store
    的构建开关与参数。

相关演进与边界文档：

- `runtime-scheduler-evolution.md`：wait、monitor/profile 统一和三个创新点的演进；
- `completion-driven-split-runtime.md`：completion protocol 与 resident Split 细节；
- `remaining-contributions-and-next-route.md`：当前缺口和下一阶段协议；
- `mc-result3-design-analysis.md`：已有 radio 结果对应的功能判断。
- `covariance_subspace_notes.md`：128-way time-frequency RFI covariance/subspace 场景、
  row/pair correlation 控制与 Celerity 对照设计。

本文只描述当前代码所实现的功能和设计边界；性能收益、置信区间 calibration、长期
profile store 稳定性以及 SplitHalo 的效果仍需要后续实验或实现验证，不能由设计描述
代替。
