# `f83d5ce` 之后的 runtime commit 审计

## 结论

`f83d5ce7c75a` 之后到当前 `2a19c9f4da8b` 一共有 18 个 commit，累计修改
13 个文件，净 diff 为 4,836 insertions / 705 deletions。它们不能被等价地写成 18 项
功能，更不能直接当成论文贡献。最终代码实际形成六条演进线：

1. private CUDA context 下的跨 context copy、queue 顺序和 Split CG 生命周期正确性；
2. Split 的 canonical merge、wide-DAG admission 和端到端计时；
3. 稳定 profile identity、monitor 快照、非阻塞 profile 传播和统一 `CostEstimate`；
4. 单 rank completion-driven ready queue；
5. 显式 accessor 合同证明的 partition-resident Split；
6. profile journal/跨 shape 学习，以及共享只读小表的版本化跨设备副本。

其中存在三类不应保留为最终设计的历史代码：已经被替代的 single-first/hysteresis、曾经
默认关闭 Split 的过渡策略、以及只为定位 CUDA 700 加入的逐 part 日志。建议论文和最终
patch series 都按上面六条线重组，而不是沿用当前 commit 边界。

以下“必要性”分为：

- **必须保留**：否则存在错误结果、悬空指针、非法 CUDA context、错误 wait 语义或
  调度协议不一致；
- **核心机制**：不是所有程序正确运行所必需，但若保留当前论文 claim 就必须存在；
- **性能支撑**：修复 prototype 架构产生的串行化或控制开销，不应单列成论文贡献；
- **可选/需消融**：可提高冷启动或可观测性，但当前参数和收益必须由实验支持；
- **已被替代/仅诊断**：不应出现在整理后的最终 patch 中。

## 逐 commit 审计

### 1. `9fa5c5973695` — `[X][Fix] pi_cuda #define SNMD alter TOBECHECK`

实际功能：

- CUDA PI 不再依赖只在 `libsycl` target 中定义的 `SNMD_OFFLINE` 宏来选择 copy primitive；
  直接比较 source/destination PI context，同 context 使用 `cuMemcpyDtoDAsync`，跨 context
  使用 `cuMemcpyPeerAsync` / `cuMemcpy3DPeerAsync`。
- 对 buffer、queue、context、offset、pitch、slice 和溢出进行显式验证，避免把错误参数
  延迟成 CUDA 700。
- handler 把完整 dim-0 row block 线性化为 1-D `Requirement`，让 Split 输入复制和输出
  merge 走连续 buffer-copy path，而不是易错的跨 context CopyRect。

必要性：**必须保留**。当前 runtime 为每设备建立 private context；若仍把跨 context
device pointer 交给 `cuMemcpyDtoDAsync`，行为无效。线性 row-block 转换也与当前
dim-0 contiguous Split 合同一致。需要注意，这是一项 CUDA/private-context 后端修复，
不是调度贡献；若未来改成共享 context，应重新评估是否还需要维护这段 plugin fork。

### 2. `616a536ab999` — `[X][Fix] CHECKED cg.hpp pi_cuda.cpp SNMD_OFFLINE`

实际功能：删除 `cloneForSplit` 周围失效的宏门控和大段旧注释实现；CUDA copy 再次明确
只由真实 context 决定，不由 `SNMD_OFFLINE` 编译宏决定。

必要性：**必须保留但应与前一 commit squash**。它没有独立研究功能，只是消除两个
target 宏配置不一致和重复实现。

### 3. `7929e7030f97` — `[X][Fix] pi_cuda.cpp`

实际功能：`RectFits` 不再把 CUDA plugin 的私有 buffer storage 类型写进 lambda 参数，
改为传入 `get_size()` 的标量值；对应的是一个编译修复，而不是算法变化。

必要性：**必须保留但应 squash 到 CUDA copy 修复**。

### 4. `f6794bba9483` — `[X][Fix] move cg.hpp cloneForSplit to cg.cpp`

实际功能：把约 150 行 `CGExecKernel::cloneForSplit` 从公共头文件移入新的 `cg.cpp`，并
加入 CMake source；clone 为 accessor requirement、argument storage、event/resource 等
建立独立所有权。

必要性：**必须保留**。Split part 不能共享会被 scheduler 重绑的 raw requirement
pointer；把实现移出头文件本身主要是可维护性/编译边界改进，应与 clone 正确性作为一个
patch。

### 5. `265cecc7a139` — `[X][Fix] split reuse pending read replica`

实际功能：记录 pending Split 的 read/write memory objects；在同一 wait window 内，
若 read-only buffer 已在目标 Split device/context 上有完整副本，则允许后继 non-Split
reader 复用，并在 write-after-read 时强制先完成相关 pending Split。

必要性：**必须保留其依赖/失效语义，复用本身属于性能支撑**。没有 WAR 检查会发生
读写竞态；没有副本复用则会把同一个只读 allocation 再写一遍并与 reader 竞态。该实现
后来被 completion cache 和 dispatch replica 逻辑扩展，最终应只保留一套统一的
versioned replica ledger。

### 6. `cf8cdaa755e9` — `[X][Fix] in_order offline queue`

实际功能：

- 每个 offline device 复用一个 private-context、in-order、profiling queue，使“一张
  GPU 是一条 HEFT processor timeline”的模型与实际 CUDA stream 顺序一致；
- CUDA buffer read/write 和 kernel mem arg 增加 context/bounds 检查，在 launch 前拒绝
  错误 allocation binding。

必要性：**必须保留**。若每次 rebind 新建 out-of-order queue，daemon 的 resource
reservation 与实际执行不一致，pending Split、copy 和普通 kernel 会在同卡意外重叠；
错误 mem arg 还可能把真正错误延迟成后续 CUDA 700。代价是 prototype 主动放弃同一 GPU
上的多-stream overlap，论文应明确当前 resource model 就是一卡一 compute timeline。

### 7. `088dd50a6df1` — `[X][Fix][Split][700] part event log`

实际功能：给每个 Split part wait 增加前后日志；同时撤回 split-to-split read replica
shortcut，只保留较保守的 non-Split reader reuse。

必要性：**仅诊断/历史回退**。逐 part 日志仅用于定位错误，正式计时必须关闭；被撤回的
复用后来由带 version/partition 语义的实现替代。整理 patch 时不应单列此 commit。

### 8. `6a6e25692c3` — `[X][Fix] Split CG life cycle`

实际功能：

- clone 对普通捕获值和 pointer argument 做 owned byte snapshot，修复 `std::function`
  target copy 后 argument pointer 悬空；
- 把 pending Split 的 wait 和 merge 分成两步，先等待所有相关 part，再入队任何 merge，
  避免 stable in-order queue 中出现“后继 kernel 已排在前面，前驱 merge 后插入”的错误
  顺序与错误归因。

必要性：**必须保留**。前半是明确的 use-after-lifetime 修复，后半是 queue 顺序和异常
传播正确性。应与 `cloneForSplit` 和 in-order queue 合成同一正确性系列。

### 9. `34b0aa89f1c6` — `[X][IMP] Split Control`

实际功能包含五部分：

1. canonical merge device：普通 Split 只在 ordered device set 的第一张卡暴露完整版本；
2. single-first：没有 single profile 前拒绝 Split；
3. profiled Split hysteresis：Split 至少快 15% 才保留；
4. wide-DAG guard：同一 DAG depth 的 task 已可填满 GPU 时优先 whole-task placement；
5. Split movement/wait 统计与 compile-time 全局禁用 Split 开关。

必要性逐项判断：

- canonical source-of-truth：**必须保留**，否则 daemon 估计的 producer location 与 handler
  实际 merge location 不一致；
- wide-DAG guard：**核心机制**，它显式保护 task parallelism 不被 gang Split 抢占；但
  当前只用 depth width，是保守 admission，而不是完整 throughput proof，必须消融；
- single-first 与 15% hysteresis：**已被替代**。后续 cold Split probe 和统一风险目标
  已接管，不能和新 selector 并存为第二套 policy；
- `SNMD_OFFLINE_TEST_DISABLE_SPLIT=1`：**只应作为消融**，该 commit 中默认启用它只是
  过渡状态；
- stats：**可选诊断**，对归因很重要，但不能打开后计入正式性能。

### 10. `d3241360419a` — `[X] runtime improve`

实际功能：

- 用 kernel symbol 的稳定 FNV-1a identity、NDRange 与 accessor shape 构造 profile key，
  修复跨 daemon/process 的 `std::hash` 不稳定和同 shape 不同 kernel 污染；
- profile 使用 Welford mean/M2/min/EWMA，profile key 在 CG 清理前捕获；
- 单 rank 新 wait window 重置历史 synthetic HEFT time；
- monitor、profile table 和全局 scheduler state 加锁/快照，monitor 只作为 service-time
  scale 消费一次；近期本程序 profile 可抑制把自己的刚结束负载再次计为 external load；
- 增加近似 memory admission；
- worker 用 `MPI_Isend` 发布 profile，master `MPI_Iprobe` drain，只在退出时 flush；
- 明确分离 user fence wait 与 timestamp query；
- Split 从 compile-time 全局关闭改成 runtime opt-in（下一 commit 又默认打开）。

必要性：

- 稳定 identity、profile lifetime、单 rank calendar rebase、锁/快照和 fence/profile
  分离：**必须保留**；
- Welford 统计和非阻塞 profile：**核心机制**，分别支撑不确定度和避免 profile barrier；
- multi-rank profile 传播：当前单节点论文范围内**非必需但实现合理**；不能把它写成
  multi-rank completion scheduling；
- fresh-profile monitor suppression、固定 service scale 和 memory estimate：**需消融/
  校准**。尤其 e236 后确认每个 Split part 仍分配 full virtual buffer，早期按 writable
  bytes/parts 估内存的版本已不成立；
- default-off Split：**过渡策略**，不是最终功能。

### 11. `ece7c73b073f` — `improve`

实际功能是本轮最大的控制面重构：

- 定义 `DISPATCH_BATCH_V1` / `COMPLETION_BATCH_V1`，completion 与 profile 解耦，
  `duration_ns=0` 仍释放资源；
- calendar/device model 改为 thread-local，batch HEFT 与 rolling dispatcher 共用统一
  candidate selector 和 transfer plan；
- 单 rank completion-driven 状态机按真实 event 完成逐波 dispatch，Split 按 gang
  reservation，传输 reservation 只在 wave 内有效；
- handler 轮询 in-flight event、批量报告 completion、传播 async error，并禁止协议启动
  后静默降级造成双重执行；
- Split profile 计入 prepare + all-parts + materialization 的端到端时间；
- 重新引入带 cache 语义的 window-local read-only replica reuse；
- 以“数据本地候选 vs 迁移候选的置信区间”实现第一版 uncertainty-aware migration；
- 用 bounded cold probe 代替 single-first，并重新默认启用 Split。

必要性：

- completion protocol、状态机、gang reservation、错误/重复 completion 校验和 handler
  event polling：**核心机制且必须保持协议正确性**；这是当前贡献二的主体；
- thread-local state：**必须保留**，否则多应用 daemon thread 会互相覆盖 calendar；
- Split 端到端计时：**必须保留**，否则 profile 会系统性偏爱 Split；
- transfer plan：**核心机制**，但有限时间 reservation 在 wave 后清除仍是模型近似，
  不是实际 copy completion；
- 第一版 uncertainty-aware migration：**已被下一 commit 的统一风险目标替代**，不应以
  post-hoc data-local veto 的形式继续存在；
- completion queue 只在 `onrun_size==1` 启用，因此不能声称已经完成 multi-rank rolling
  scheduling。

### 12. `bcc7fa75f955` — `CostEstimate`

实际功能：把 cold、exact profile、capability-scaled profile、derived Split 全部改为
`CostEstimate{mean, uncertainty, samples, source}`；execution 与 transfer uncertainty
通过 `hypot` 传播，所有 Single/Split candidates 使用同一个
`finish + beta * uncertainty` 风险目标和确定性 tie-break；删除独立 Split hysteresis 与
post-hoc migration veto，并让 HEFT upward rank 也消费风险调整后的估计。

必要性：**核心机制**。如果论文声称“统一、不确定性感知的联合决策”，这是必须保留的
实现边界；否则系统仍只是多个 heuristic 串联。以下部分仍必须实验验证：

- cold/scaled/derived/transfer 的 50%/20%/30%/15% 先验；
- `beta=1.96` 是否在不同设备和 kernel family 上校准；
- `hypot` 隐含的误差独立性；
- 用 sample standard deviation 预测下一次 service time 是否优于 MAD/quantile；
- 风险目标提高稳健性时牺牲了多少平均 makespan。

### 13. `e236e3a6e37a` — `cele`

实际功能与 commit message 无关，主体是 partition-resident Split：

- 新增逐 accessor 的 `handler::ext_snmd_partition_local()` 显式合同；
- protocol/profile key 增加 access range、offset、sub-buffer、partition-local 与
  persistent mode；
- daemon 只有在所有 writable accessors 合同成立、shape/atomic/sub-buffer 等约束通过且
  存在兼容 successor 时才生成 resident candidate；
- handler 对真实 CG 再验证，失败降级为 ordinary materializing Split；
- 同 device scheme 的兼容 producer-consumer edge 复用 partition，incompatible edge
  canonicalize；旧 logical version 可被新 resident write supersede；
- ordinary Split 与 resident Split profile 分 key，resident producer profile 不重复计入
  延后发生的 materialization；
- memory admission 修正为每 part 仍需 full virtual allocation，而不是 write bytes/parts。

必要性：**贡献三的核心机制，同时安全检查必须保留**。没有逐 accessor 合同就无法从
普通 SYCL accessor 推断间接索引/halo/global read；没有 handler 二次验证就会把 daemon
metadata 错误升级成内存错误。它不是通用 virtual buffer：当前只支持单 rank、dim-0
contiguous、同一 wait window，并仍在每设备分配完整虚拟 buffer。API 中对 accessor
内部对象的桥接是 experimental fork surface，需要专门的 compile/ABI test。

### 14. `dbc5f5344cf5` — `profile journal`

实际功能不只是 journal：

- 后台线程异步 append `SNMD_PROFILE_OBS_V2`，支持 namespace/path/age、bounded queue
  和 V1/V2 解析；
- daemon 启动时把历史样本加载为 `persisted_profile`，并给额外 uncertainty floor；
- 从 identity/structural shape neighbors 学习 cold estimate，区分 exact、persisted、
  learned identity 和 learned structural source；
- feature 包含 precision、NDRange、bytes、access mode、partition-local 和 persistent
  Split，避免不同 mode 混学。

必要性：

- journal 若要主张“跨 daemon 冷启动”则是**核心机制**；后台写线程和 bounded queue 是
  保证 completion path 非阻塞所必需；
- learned neighbor transfer 是**可选/需强消融**，并非基本调度正确性所需。20%/60%
  identity/structural error 目前仍是人工先验；
- 默认持久化开启会污染独立实验，正式结果必须设置 `SYCL_SNMD_PROFILE_PERSIST=0`，或为
  每个参数/二进制使用独立 namespace；
- store 是 append-only，90 天 age 只控制加载，不控制文件无限增长，artifact 前应增加
  compaction/size cap 或清理工具。

### 15. `2d865ec242d3` — `[X] Fix cg compile_error`

实际功能：修正 `cloneForSplit` 中类型/构造细节导致的编译错误。

必要性：**必须保留但应 squash 到 e236/CG clone patch**，没有独立功能价值。

### 16. `5d766015e548` — `[X] Fix handler compile_error`

实际功能：调整 partition-local API 对 incomplete internal accessor type 的使用，把完整
类型相关验证留在 source 实现侧。

必要性：**必须保留但应 squash 到 resident contract patch**。

### 17. `4c4b1e708eae` — `adaptive fix`

实际功能：针对 private context 下 DPC++ 每个 `MemObjRecord` 只有一个 `MCurContext` 的
限制，在一个 daemon dispatch 开始前识别共享 constant-like inputs，并为所有 reader
device 预建副本。准入条件是：本 wait batch 只有 full-buffer read、当前 dispatch 至少
两张设备读取、非 sub-buffer、对象默认不超过 65,536 bytes；direct D2D 失败可回退
D2H→H2D。后继 Single/Split reader 使用 cache，write 继续失效。

必要性：**性能支撑，且对当前 private-context 架构实际上必需**。没有它，多个独立
chemistry task 共享 2 KiB coefficient table 时，普通 scheduler 会把 read/read 表示成
一次 read_write migration，并等待前一设备 reader，正确的 HEFT placement 被 host
submission 串行化。它修的是数据层假依赖，不是新的调度算法。准入必须保持保守；若扩展
到局部 read 或大对象，需要 region/version tracking，不能只提高 size threshold。

### 18. `2a19c9f4da8b` — `34 fix`

实际功能包含两项：

1. 在 `MemObjRecord` 增加单调 `MWriteVersion` 和 weak lifetime handle，使 full-read
   replica 可跨 wait 保留；任何 read_write/discard/atomic submission 推进 version，
   stale record/version 的 cache entry 立即失效；环境变量可关闭跨 wait persist。
2. cold Split 最小 single-cost gate 前移到 device-subset enumeration、transfer plan 和
   Split estimate 之前，避免 completion dispatcher 对注定拒绝的候选重复做昂贵工作。

必要性：

- version + weak record：**必须保留**，否则跨 wait cache 会形成 use-after-lifetime 或
  stale read；它把前一 commit 从 window-local 优化变成安全的 steady-state 优化；
- early reject：**性能支撑**，不改变候选集合，但 adaptive 日志中每次运行上千次 cold
  reject，前移能直接减少控制面 dispatch bubble；
- 当前 version 是 whole-buffer 粒度，安全但保守；未来 partial writes 只能失效整个
  replica，不能把它描述成通用 region version ledger。

## 推荐的最终 patch series

为论文 artifact 和后续 review，建议把 18 个 commit 重排为以下 6 个可独立测试的 patch：

1. **Private-context Split correctness**：`9fa5 + 616a + 7929 + f679 + cf8c + 6a6e`
   以及 compile fixes；覆盖 CUDA peer copy、stable in-order queue、CG deep clone/lifetime。
2. **Split source-of-truth and admission**：保留 `34b0` 的 canonical merge、wide-DAG guard
   与 stats，删除 single-first、独立 hysteresis 和默认 hard-disable。
3. **Stable uncertain cost model**：`d324` 的 identity/statistics/snapshot + `bcc7` 的统一
   `CostEstimate`；把所有先验集中为可运行时配置并提供敏感度实验。
4. **Completion-driven dispatch**：`ece7` 的协议、状态机、rolling selector、event polling
   与 failure semantics；单独注明 single-rank capability gate。
5. **Contract-proven resident Split**：`e236` + 两个 compile fix；普通 materialize 和
   resident mode 分开测试。
6. **Adaptive knowledge/data reuse**：`dbc5` 的 journal/learned cold start，与
   `4c4b + 2a19` 的 versioned read replica 分成两个逻辑子模块，分别做 A/B。

## 必须补的验证

当前历史主要由应用日志驱动，若要把“必要”从代码审计升级为证据，还需要：

- CUDA backend unit test：same-context 1-D/rect copy、cross-context peer copy、无 P2P
  fallback、invalid pitch/offset/context；
- Split clone test：scalar、pointer、accessor、lambda/std::function 捕获在原 CG 销毁后仍
  正确；
- protocol fault injection：duplicate/unknown completion、zero duration、part failure、
  daemon failure after first dispatch；
- replica test：read/read 跨 4 contexts、下一 wait reuse、device/host/atomic write 后失效、
  MemObjRecord 销毁与地址复用；
- resident contract negative test：offset、sub-buffer、atomic、global/halo read、device
  scheme 改变、incompatible successor；
- 风险模型 sensitivity：所有 uncertainty percentage、`beta`、monitor fresh window、
  journal on/off、identity-only/structural learning；
- 正式计时关闭三类 trace，清空/隔离 profile store，并同时报告 native 1 GPU、系统
  1/4 GPU、Celerity 1/4 GPU。
