# 从 Krylov 到 Reactive Transport Ensemble：Split 正确性与性能演进

本文记录 `reactive_transport_ensemble_sycl_timer.cpp` 的设计来源、它为什么曾被选作 Krylov 之后的正面案例候选，以及它在当前运行时 Split 路径上的正确性适配、CUDA 700 排错和性能归因。`8192×8192, 100 steps` 已经稳定完成；第一轮调度修正又把持续 Split 的 `92.194035 s` 降到默认策略的 `11.459888 s`。这证明 TB 级 Split 复制确实是旧结果的主因，但新结果仍比原生单 GPU 的 `10.154259 s` 慢 `12.86%`，所以 Reactive 目前是稳定的正确性、Split admission 和 whole-kernel placement 压力案例，还不是系统的正面加速结果。

`reactive_transport_ensemble_notes.md` 主要描述最终 miniapp 的科学语义、Celerity 映射和实验假设；本文更关注“为什么这样演进”和“运行时为了保持原 Split 设计而修了什么”。

## 为什么不再把 Krylov 当作正面案例

原 GMRES/Arnoldi 版本很真实，但它同时包含：

- basis 串行递推；
- dot/norm reduction；
- host scalar decision；
- Hessenberg 小矩阵；
- read_write 和重复使用的临时量；
- restart 和大 fan-in。

它对 runtime 是很好的压力测试，却不是一个容易归因的正面案例。Reduction、标量同步、CG clone、split merge、跨设备数据一致性和消息大小中的任何一个问题都可能导致卡住或 CUDA 700。

为了绕开这些问题构造的 Krylov Poly 又走向了另一个极端：每个 cycle 约 92 个偏小 kernel，虽然有短 fan-out，但很快 fan-in 回到下一 basis，逻辑并行宽度持续时间短。实测中 profile、daemon 通信、queue rebind 和 event 查询约增加 0.2 ms/kernel，几万枚小 kernel 无法摊薄这些成本。

因此 Krylov 最适合保留为负面案例：它说明细粒度、强依赖、频繁 profile 的 workload 可能只有调度开销，没有可兑现的多 GPU 并行收益。

新的正面案例需要同时满足：

- kernel 数量适中而单 kernel 足够重；
- 一个 wait window 内有持续的逻辑并行；
- 同时存在“整 kernel 放置”和“大 kernel Split”两种选择；
- 多个阶段成本不同，且多个 GPU 的速度可以不同；
- 重复物理时间步可以让 profile 收敛；
- 写集合天然是 dim0 row block，符合当前 Split merge 模型。

Reactive Transport Ensemble 正是按这些条件设计的候选。后面的实测说明：它满足 kernel 粒度和表面 DAG 宽度要求，却进一步暴露了一个设计范式中必须补充的条件——当任务级并行已经足够填满设备时，不能再让大量通信密集型 kernel 持续占用全部设备做 Split。

## Miniapp 的科学计算逻辑

每个 ensemble member 表示一个具有不同网格精度、反应速率和化学子步数的二维反应输运场。每个网格点有三种 species。

一个完整物理时间步采用 Strang splitting：

```text
state[e,t]
    |
    v
R[e]  Robertson reaction(dt/2)
   / \
  v   v
X[e] Y[e]  WENO5 x/y flux divergence
   \ /
    v
F[e]  transport update + Robertson reaction(dt/2)
    |
    v
state[e,t+1]
```

四类 kernel 的数据访问如下。

### R：第一半步刚性化学反应

读取：

- `state.a`；
- `state.b`；
- `state.c`。

写入：

- `chemistry.a`；
- `chemistry.b`；
- `chemistry.c`。

每个 work-item 在自己的 cell 上执行若干次展开的 3×3 Rosenbrock-Euler 线性求解。不同 member 的 `chem_substeps` 不同，因此相同访问模式不代表相同计算成本。

### X：x 方向 WENO5

读取：

- 三个 `chemistry` species；
- `velocity_x`。

写入：

- 三个 `flux_x` field。

每个 work-item 为三个 species 各做左右两个界面的五阶非线性重构。它读取当前 row 上半径 3 的列邻域。

### Y：y 方向 WENO5

读取：

- 三个 `chemistry` species；
- `velocity_y`。

写入：

- 三个 `flux_y` field。

它与 X 没有写冲突，读取的是相同 chemistry 版本，但 stencil 沿 row 方向。因此 `X[e]` 和 `Y[e]` 是真实的同层独立分支。

### F：输运汇合和第二半步反应

读取：

- 三个 `chemistry` field；
- 三个 `flux_x` field；
- 三个 `flux_y` field。

写入：

- 下一时间步的三个 state field。

它先做九输入的 transport fan-in，再执行第二个 Robertson reaction half-step。

### 依赖和并行宽度

单个 member 内只有以下 RAW 依赖：

```text
R -> X
R -> Y
R -> F
X -> F
Y -> F
```

不同 member 之间没有依赖。对 `E` 个 member，一个 full-mode window 有 `4E` 个 coarse kernel，最大宽度为 `2E`，关键路径只有 `R -> max(X,Y) -> F` 三层。

默认 `E=6` 时，每个时间步只有 24 个 kernel，但最大逻辑宽度是 12。这和 Krylov Poly 的“约 92 个小 kernel、反复短 fan-out 后立即 fan-in”完全不同。

## 为什么 kernel 本体足够大，但仍覆盖不了错误的 Split 决策

R 和 F 中，每个 cell 都执行多次展开的刚性 ODE 小求解；X/Y 则包含 WENO smoothness indicator、非线性权重和六次重构。它们不是简单的 copy 或 axpy。

默认六个 member 的 chemistry 子步数为 16、18、20、22、24、26。默认网格总计 14,863,584 个 cell：

- 一层 R 执行 290,147,904 个 cell-substep；
- R 与 F 的两个 half-step 合计 580,295,808 次展开 Rosenbrock solve；
- X/Y 合计 178,363,008 次非线性 WENO reconstruction。

`--nx 8192 --ny 8192 --steps 100` 的日志中，17 个 field/member 的总数据估计达到 16,193,842,048 bytes。这个参数是 runtime、显存和跨设备移动的压力配置，不应与默认正确性配置混为一谈。

原生单 GPU 实测 `run_sec=10.154259`，共 2400 个逻辑 kernel，平均约为 `4.23 ms/kernel`。这已经显著大于 Krylov Poly 中测得的约 `0.2 ms/kernel` 固定调度/profile 增量。因此旧 `92.194035 s` 结果的主要问题不能归为“kernel 太小，无法覆盖固定 profile 开销”；当时真正没有被覆盖的是每次 Split 的完整 read replica 和 owned-output merge。禁止无收益 Split 后剩余的约 `1.31 s` 差距则需要重新拆分为调度/profile 控制面、普通 whole-kernel 数据移动、设备有效速度和负载粒度，不能继续沿用旧归因。

## 为什么一个时间步是正确的 wait window

程序只在完整物理时间步结束时 `queue.wait()`。这不是为了绕过消息大小或每 N 个 kernel 强制切图，而是时间积分的真实语义边界：下一步必须读取已经完成的 `state[e,t+1]`。

因此：

- step 0 使用 cold-start cost；
- step 1 以后可以使用上一窗口的 device/split profile；
- R、X、Y、F 之间没有人为 wait；
- `--wait-each-kernel 1` 只用于定位问题，不属于性能运行。

## Miniapp 的内存和 Split 语义

每个 member 有 17 个 row-major 2D field：

- 两组 ping-pong state，共 6 个；
- chemistry，共 3 个；
- flux-x，共 3 个；
- flux-y，共 3 个；
- velocity-x/y，共 2 个。

所有 kernel 的 NDRange 都是 `(rows, cols)`，所有写 accessor 都覆盖完整 `(rows, cols)`。沿 NDRange dim0 切分后，每个 part 写一个完整、连续、互不重叠的 row block。

当前 bench 使用 read + discard_write 的 out-of-place 版本，是为了让 producer version 和 DAG 容易审计，不是因为系统不能 Split `read_write`。

`read_write` 的正确 Split 合同是：

1. 每个 part 在执行前得到完整的 pre-kernel read snapshot；
2. work-item 按 dim0 分区；
3. 每个 part 只写自己的 owned row block；
4. merge 只写回该 part 的 owned block，而不是整份 private replica。

R 如果改成 cell-wise in-place，可以满足这个合同。X/Y 若一边读邻居一边覆盖同一个输入 field，则不能只凭 owned write range 证明安全；它还需要显式 snapshot 或 double buffering。

运行时能够从 accessor 判断形状、mode 和数据范围，但不能从一个 `read_write` mode 自动推断真实写集合是否重叠。因此“partition-disjoint write”仍是系统与 kernel 之间的语义合同。

## 原始 Split 核心设计

Reactive 接入前保留的 Split 核心思想并不是“把一个 kernel 分成两份，然后立即全部 wait”。它的流程是：

```text
daemon 给出 NumParts 和 SplitDevices
        |
        v
handler 为各 part 选择 queue
        |
        +-- 为 read/read_write 准备完整输入 replica
        +-- 为 write output 准备每设备 allocation
        |
        v
GraphBuilder clone CG，并修改每个 part 的 GlobalOffset/GlobalSize
        |
        v
各 GPU 异步执行 split sub-kernel
        |
        v
handler 保存 part events、read set、write set 和 owned merge ranges
        |
        +-- 无关 kernel 可以继续提交
        |
        v
真实 consumer 到来或 batch 结束时 wait + merge owned rows
```

这里需要区分两种“异步”：

- Split kernel 的执行和结果 merge 是延迟的。只有真正读取 pending write、或者覆盖 pending read version 的 kernel 才触发 finalize；无关 kernel 可以继续提交。
- 当前 input preparation 还不是完全异步。D2D 或 D2H→H2D copy 后存在 host-side event wait，确保 sub-kernel 启动前 replica 已经可用。

所以原设计真正保留的性能核心是“异步 sub-kernel + dependency-driven deferred merge”，而不是“所有数据移动都不等待”。后续修正没有取消这个核心。

## 第一阶段：让 daemon 知道什么 Split 是合法的

最初 daemon 主要看到 accessor 数据量，缺少完整 NDRange 和访问范围信息，容易把“数据量很大”误当成“可以按 dim0 连续切分”。从 `be368354` 开始增加了：

- `work_dim` 和三维 global size；
- daemon 消息中的 accessor element size 和 memory range，以及 handler 本地保留的 access range；
- profile key 中的 NDRange shape；
- daemon 与 handler 两侧的 Split legality check；
- Split profile 使用包含准备和 merge 的 host wall time，而不是只看一个 part 的 device event。

最终 handler 只接受当前实现能够正确 merge 的布局：

- `global_size0` 能被 part 数整除；
- 写 accessor 的 `AccessRange[0]` 与 kernel dim0 对应；
- trailing dimensions 覆盖完整 row；
- trailing offset 为 0；
- 不接受 sub-buffer；
- 每个 owned row block 在 row-major buffer 中是一个连续区间。

这类检查属于正确性边界。它通常只增加极小的 host 判断成本，但会拒绝原先可能被错误 Split 的候选，因此可能让某些 kernel 退回单 GPU。退回不是性能回退 bug，而是当前 Split 能力边界的真实体现。

## 第二阶段：从整 buffer 假设改成真实 accessor region

早期 merge 按 `MMemoryRange` 切分，默认 kernel range、access range 和完整 buffer range 完全相等。这在 offset accessor、partial accessor 或 sub-buffer 下会计算错误的 copy offset。

修正后：

- partition 大小来自 `MAccessRange`；
- offset 保留原 accessor 的 `MOffset` 和 `MOffsetInBytes`；
- 完整 row block 被线性化成 1D transfer requirement；
- 线性 offset、元素数和 byte offset 都做 overflow/bounds 检查；
- merge 仍然只复制 owned write block。

把连续二维 row block 描述成 1D copy 还有一个现实原因：CUDA 的跨 context 2D CopyRect 需要处理 origin、pitch、slice pitch 和 context，出错面远大于连续 buffer copy。

这一阶段主要是边界正确性修正。1D copy 也可能比 CopyRect 路径更直接，但它不是算法层的性能优化。

## 第三阶段：Split CG 必须拥有独立、完整的生命周期

原 `cloneForSplit` 曾只复制 `MRequirements`、`MArgs` 等容器，但容器内部仍有指向原 CG 的 raw pointer。随着原 CG 被 move、销毁或其内存被复用，会出现两类悬空：

- accessor argument 和 accessor 的 range/offset 参数仍指向原 `AccessorImplHost`；
- scalar argument 仍指向原 lambda 的捕获存储。

修正分了几步：

1. 深复制 `AccessorImplHost`，建立 old requirement 到 new requirement 的映射；
2. 重映射 accessor 本体、access range、memory range 和 offset 参数；
3. 重映射显式 `MArgsStorage`；
4. clone `MHostKernel` 并尝试映射 HostKernel 内部参数；
5. 最终对 `kind_std_layout` 和 `kind_pointer` 做值快照，让每个 split clone 自己拥有参数 bytes。

最后一步是 result7 后定位到的关键问题。`item/id/nd_item` kernel 会通过 `std::function` 规范化，`extractArgsAndReqsFromLambda` 得到的 `rows/cols` 指针可能位于 `std::function` 的 heap target，而不在 `HostKernelBase::getPtr()/getSize()` 表示的对象本体内。仅做地址区间映射仍会留下悬空 scalar。

WENO-Y 用捕获的 `rows` 计算 `rp1/rp2` 边界。如果该值在原 CG 释放后被覆盖，最后几行 work-item 就可能访问 row 4096 或更远位置，从而真正产生 CUDA illegal address。普通值参数快照修复了这个生命周期缺口。

这完全属于正确性修正，对 device kernel 性能没有影响；host 侧每次 Split 会多复制少量 kernel 参数 bytes，成本可忽略。

## 第四阶段：allocation 必须由 context 和 device 共同标识

早期 Split 通过 `MemObjRecord::MCurContext` 选择当前 allocation。多 GPU 共用一个 platform default context 时，同一个 context 内可能有多个设备 allocation，只按 context 查找会把 GPU A 的 CUdeviceptr 绑定给 GPU B 的 kernel。

这类错误通常不会在 setArg 处立即暴露，而会在异步 kernel 运行后污染 CUDA context，随后在任意 event wait 或 copy API 上报告 700。

修正包括：

- 每个 offline device 使用一个稳定 private context；
- Split allocation 查找同时匹配 context 和 device；
- memory move 接收真实 source queue，而不是只有 source context；
- CUDA kernel mem argument 在 launch 前验证 allocation context 与 kernel context；
- buffer read/write/copy 检查 queue、mem object、context、offset 和 size。

这些修改把“晚到的 CUDA 700”尽量转成“早到的 PI invalid context/value”。

这是 correctness 与 performance 都受影响的修正。Private context 消除了 allocation 身份歧义，但跨 context D2D 必须走 CUDA peer API；如果硬件、拓扑或 context 不允许 peer access，就只能 fallback 到 D2H→H2D。

## 第五阶段：跨 context copy 和 fallback

`pi_cuda.cpp` 中的 Split 支持原来被本地 `SNMD_OFFLINE` 宏包裹。CUDA plugin 是独立 target，不能依赖 `source/detail/daemon/define.hpp` 的宏自然传播。若宏未定义，运行时会把跨 context pointer 送进普通 `cuMemcpyDtoDAsync`，这是非法的。

最终设计是：

- `cg.hpp/cg.cpp` 的 `cloneForSplit` 始终编译；
- CUDA buffer copy 根据实际 PI context 选择普通 D2D 或 peer copy；
- 这些 capability 不再由 `SNMD_OFFLINE` 宏决定是否存在；
- 是否真的 Split 仍由 source 层的调度 policy 决定；
- source 文件中的 `#ifdef SNMD_OFFLINE` 保留，用于标识和控制调度路径。

跨 context 连续 buffer copy 使用 `cuMemcpyPeerAsync`，CopyRect 使用 `cuMemcpy3DPeerAsync`，并在调用前验证 origin、pitch、slice pitch 和 buffer bounds。

如果 peer access 不可用，PI 返回 `PI_ERROR_INVALID_OPERATION(-59)`，handler 捕获后走：

```text
source GPU -> host staging -> destination GPU
```

这条 fallback 保证正确性，但对性能影响很大：它增加两次传输和 host-side wait。result5–7 中频繁出现的 `-59` 并不是 CUDA 700，而是当前机器上 direct peer path 不可用的明确信号。

## 第六阶段：pending read/write version 不能只看 write output

最初 `kernelTouchesPendingOfflineSplit` 只检查后续 kernel 是否读取 pending split 的 write output。Reactive 的 X/Y 两个分支还暴露了另一种 hazard：多个 pending kernel 可以共同读取同一 chemistry version，而后续 kernel 可能覆盖该 buffer。

因此 pending state 增加：

- `ReadMemObjs`；
- `WrittenMemObjs`；
- part event 列表；
- 每个 part 的 owned merge requirement。

现在触发 finalize 的条件包括：

- 后续 kernel 触及 pending write；
- 后续 kernel 要写 pending kernel 仍在读取的对象。

曾经尝试让后一个 Split 直接复用前一个 pending Split 已准备的 read-only replica。这能减少 X/Y 对 chemistry 的重复复制，但现有 `MemObjRecord` 只有单一 `MCurContext`，没有完整的 per-device version/replica epoch。仅凭“这个设备上存在一个 replica”无法证明它对应当前 producer version，也无法证明没有 transfer stream 正在覆盖它。

所以当前采取较保守的策略：

- split-to-split 不跳过 version preparation；
- non-split read 若明确命中 pending read-only replica，可以复用；
- 写 hazard 仍会触发 finalize。

取消 split-to-split replica shortcut 是明显的性能代价，尤其 X/Y 同时读取三份 chemistry 时会增加传输。但在建立真正的 per-device version table 之前，这是必要的正确性收紧。

## 第七阶段：queue 的并发必须和 HEFT 资源模型一致

早期每次 rebind 都创建新的 profiling queue，而且 queue 可能是 out-of-order。daemon 的 HEFT 却把每个 GPU 建模为一条 `available_time` 时间线。

结果是：HEFT 认为同一 GPU 上的 kernel 已按时间线串行排列，runtime 却把它们发到互不相关的 CUDA stream 并发执行。一个 pending split kernel、下一 kernel 的 input preparation 和更早 kernel 的 merge 可能发生模型外重叠。

当前每个 offline device 复用一个稳定的 profiling + in_order queue：

- 同一 GPU 上遵守 HEFT 的单时间线；
- 不同 GPU 仍可以并行；
- 一个 Split 的不同 part 仍在不同设备并行；
- 不同 member 放到不同 GPU 仍有 task parallelism。

这是 correctness 与 performance 同时受影响、而且性能影响比较明显的修正。它关闭了同一 GPU 内原本可能存在的多 stream overlap，但那些 overlap 没有被调度模型计价，也没有被依赖/version 系统完整约束，不能当成可靠收益。

未来如果希望恢复同 GPU overlap，正确方向是让 daemon 把 compute stream、transfer stream或多 queue 当成显式资源，而不是重新创建无状态 out-of-order queue。

## 第八阶段：先等待相关 pending kernel，再开始任何 merge

稳定 in_order queue 带来了一个更隐蔽的 finalize 顺序问题。假设同一设备 queue 上已经提交：

```text
kernel 137
kernel 138
```

consumer 144 同时依赖二者。旧 finalizer 按 pending vector 顺序执行：

```text
wait 137
merge 137
wait 138
merge 138
```

由于 merge 137 也是向同一 in_order queue 插入 copy，它实际排在已经提交的 kernel 138 后面。如果 138 产生 asynchronous CUDA error，错误就会在 merge 137 的 D2H API 上被观察到，看起来像 137 的 merge 越界。

现在改成两阶段：

```text
wait 137
wait 138
merge 137
merge 138
```

batch-end finalization 也先 wait 所有 pending，再执行任何 merge。

这没有把所有 Split kernel 都立即串行化。只有当前 consumer 真正涉及的 pending 集合需要完成；无关 pending kernel 仍保留。对同一 consumer 来说，它本来就必须等全部输入，所以两阶段 wait 通常不增加其语义关键路径，主要影响是避免在未确认的 kernel 后插入 merge，并让错误归因到真实 kernel/part。batch end 先 wait 全部再 merge 可能减少早期 merge 与较晚 kernel 的 copy/compute overlap，因此仍应在性能时间线中单独检查。

## result1 到 result8 的定位过程

### result1：大参数首次稳定复现

`--nx 8192 --ny 8192 --steps 100` 在约 kernel 114 附近出现 CUDA 700，错误主要在 event wait 处报告。此时只能知道某个更早的异步 kernel 已经污染 context，不能把 wait 行号当作越界源头。

### result2：同一规模仍出现大量 700

增加 CG/accessor remap 和 Split offset 修正后，错误仍在一批 wait 中级联出现。这说明单纯修正 requirement raw pointer 还没有覆盖全部 kernel 参数生命周期。

### result3：较小规模也会晚发失败

`--nx 4096 --ny 4096 --steps 50` 也在两百左右 kernel 后失败，而默认小参数能够通过。它排除了“8192 固有 stencil 边界错误”，更像是执行次数、异步状态或生命周期累积问题。

### result4：context/allocation 修正后运行更远

在 private context、context+device allocation lookup、线性 row copy 和 CUDA context validation 后，运行到 200 多 kernel。错误出现在 kernel 223 merge 的 D2H 行，但 kernel 224 已经提交，因此该行仍可能只是下一个异步 kernel 的错误观察点。

### result5：在 113/114 对上复现

同样模式又在 WENO-X/WENO-Y 成对提交后复现。它说明问题不是某个固定绝对 kernel_count，而与 DAG 中相邻的 Split kernel 和异步 finalize 顺序有关。

### result6：700 在 `pi_cuda.cpp:2794` 被观察

新增 read/write/copy 参数检查后，700 仍在 `cuda_piEnqueueMemBufferRead` 被观察。由于 offset、size 和 context 已在调用前验证，这更支持“D2H API 收到了更早 kernel 的 asynchronous error”，而不是该 read 自身越界。

### result7：part event 日志锁定真实区间

result7 显示：

- kernel 137 part 0 wait 成功；
- kernel 137 part 1 wait 成功；
- kernel 137 整体完成；
- kernel 138 已经提交；
- 随后 merge 137 的 D2H 首次观察到 700。

因此 kernel 137 和它的 part 写范围不是直接故障源。真正需要检查的是排在该 copy 前面的 kernel 138，即这一窗口中 member 5 的 WENO-Y。

代码检查随后确认了 `std::function` heap target 中 scalar capture 未被 clone 所有的问题，解释了为什么 WENO-Y 的 `rows` 边界会在运行若干窗口后损坏。同时 finalizer 改成两阶段 wait/merge，保证今后的日志把错误归到实际 kernel/part。

当时这一生命周期修正已经进入代码，但还缺少目标机长运行。最新实测现已补上该验证。

### result8：8192×8192、100 steps 完整通过

最新目标机运行完成全部 100 个物理时间步、2400 个逻辑 kernel，并给出：

```text
TIMING run_sec=92.194035
TIMING host_sec=0.422099
RESULT checksum=44.315289 ...
VERIFY passed=1
```

这说明从 result1 开始围绕 accessor/CG 生命周期、context/device allocation、跨 context copy、stable in-order queue 和两阶段 wait/merge 的修正，已经通过此前最容易复现 CUDA 700 的长压力配置。这里可以得出“该配置正确完成”的结论，但不能仅凭一次运行证明所有 accessor 形状和所有 Split kernel 都已经无边界问题。

## 第一轮性能结果：不是逻辑宽度不足，而是持续 Split 的代价

同一参数下的原生单 GPU 与第一轮持续 Split 双 GPU 系统结果是：

| 指标 | 原生单 GPU | 第一轮持续 Split 双 GPU |
|---|---:|---:|
| `run_sec` | 10.154259 s | 92.194035 s |
| 相对原生 | 1.00× | 9.079× |
| 2400 kernel 平均时间 | 4.23 ms | 38.41 ms |
| 额外时间 | - | 82.039776 s |

两次初始化分别约为 51.95 s 和 51.89 s，几乎相同，而且都不在 `run_sec` 内。因此 9.079× slowdown 不能归因于数组初始化。系统运行最终 `VERIFY passed=1`，所以这是正确结果下的数据路径和调度代价，而不是 CUDA 700 重试或错误退出造成的假慢。

默认 `members=6` 时，DAG 最大宽度为 12；机器只有两个 GPU。仅从逻辑宽度看，已经有足够多的独立 R、X/Y 和不同 member kernel 填满两台设备。GPU 利用率持续不满也不能反推出“宽度只有 1”：当每个逻辑 kernel 都占用两台 GPU 做 Split 时，完整 replica copy、host-side copy wait、part kernel 和 merge 会在时间线上形成空洞，同时本来可以放到两台 GPU 并行执行的两个独立 member 被一个 Split kernel 同时占住。

### 一阶数据量模型

设六个 member 的总 cell 数为：

```text
Σ cells = 238,144,736
one field = 238,144,736 × 4 = 952,578,944 bytes
```

对于 2-way Split，若 secondary GPU 在每个 kernel 前都需要完整 read snapshot，并在结束后把自己的一半输出合并回主完整版本，则每个时间步的近似内部流量为：

| Kernel | 完整 read replica | secondary owned merge | 合计 field-equivalent |
|---|---:|---:|---:|
| R | 3 | 3/2 | 4.5 |
| X | 3 species + 1 velocity | 3/2 | 5.5 |
| Y | 3 species + 1 velocity | 3/2 | 5.5 |
| F | 3 chemistry + 3 flux-x + 3 flux-y | 3/2 | 10.5 |
| 每步 | - | - | 26 |

于是 100 步的一阶上界为：

```text
952,578,944 × 26 × 100 = 2,476,705,254,400 bytes
                         ≈ 2.48 TB (decimal)
```

用额外的 82.039776 s 反推，等效传输率约为 30.2 GB/s，和双 GPU PCIe/P2P 数据路径的量级高度一致。这不是精确硬件计数：velocity 等真正只读 replica 若被安全保留可以减少流量，P2P 与 staged fallback 也有不同带宽。但数量级和实测 slowdown 的吻合足以把“持续完整复制”列为首要假设，后续应使用实际 byte/copy counters 验证。

### 为什么 profile 没有自动停止 Split

当前代码中有两个会让首次 Split 决策自我强化的具体原因：

1. `estimateSplitInternalCopyCost` 使用 `totalReadBytes - dependentReadBytes`，把 dependent reads 交给 dependency communication model 计费；但该模型又把 Split predecessor 的全部 `split_devices` 都当作可提供完整版本的 source。当前实现实际上只保证最终 merge device 有完整版本，其他 producer allocation 通常只有自己写过的 owned rows。X/Y 的 chemistry 和 F 的九个输入仍需要为其他 consumer part 构造完整 snapshot。修正时应先显式确定或回传 canonical merge device，再明确只由一个模型计费：要么 dependency model 只把该 merge device 当作完整 source，要么 internal-copy model重新计入这些 read bytes，不能漏算也不能双算。
2. profile table 按 `num_parts` 分桶。持续采到的是 Split end-to-end wall time，而相同 key 的单设备候选可能一直只有 cold heuristic，没有真实反事实样本。特别是非 root 多 buffer kernel 的 cold arithmetic-intensity heuristic 可能远大于真实单 GPU 时间，调度器就会继续认为已经很慢的 Split 仍优于一个从未测过的 single candidate。

此外，逐节点 earliest-finish 的 HEFT 贪心选择没有显式保护宽 DAG 的 task parallel slack。对两个相近且独立的 kernel，把第一个拆到两张卡可能得到更早的单任务完成时间，却可能比两张卡各执行一个完整 kernel 的整体 makespan 更差。这个问题属于 Layer B 的资源分配，而不是 Layer C 的 copy 实现。

因此第一轮结果的归因是：

- 正确性：长运行通过，Split 生命周期修正有效；
- Layer A：Reactive kernel 已经足够大，固定 profile 开销不是主因；
- Layer B：Split admission、反事实采样和宽 DAG 资源机会成本没有被正确表达；
- Layer C：一旦错误地持续 Split，完整 replica/merge 把代价放大到 TB 量级。

## 第二轮性能结果：Split 已受控，但双卡尚未加速

`reactive-imp-result.txt` 对同一个 `8192×8192, 100 steps` 配置给出了三组消融：

| 配置 | `run_sec` | 相对原生单 GPU | 结果 |
|---|---:|---:|---|
| 强制 `TEST_DISABLE_SPLIT`，统计关 | 11.512704 s | 1.134× | `VERIFY passed=1` |
| P1+P2+P3 默认策略，统计关 | 11.459888 s | 1.129× | `VERIFY passed=1` |
| 默认策略，`SPLIT_STATS` 开 | 11.813451 s | 1.163× | `VERIFY passed=1` |

相对旧 `92.194035 s`，默认策略快 `8.045×`，运行时间下降 `87.57%`。默认策略和强制禁止 Split 只差 `-0.46%`，属于这组单次测量的噪声范围；详细统计版本比无统计默认版本慢 `3.09%`，说明逐 kernel 输出 2400 行决策并非完全免费，正式计时仍应关闭它。

### 为什么两个 Split 开关不再改变结果

统计运行的 100 个 window 都报告：

```text
SNMD_SCHED_STATS kernels=24 single_kernels=24 split_kernels=0
SNMD_SPLIT_STATS ... split_kernels=0 ... all split copy/wait counters=0
```

所以整次运行实际选择的是：

```text
single kernels = 2400
split kernels  = 0
```

这使两个现象都可以直接解释：

- `TEST_DISABLE_SPLIT` 没有进一步改变执行路径，因为默认策略已经没有选择 Split；
- `SPLIT_STATS` 只改变观测和输出，不改变调度语义，但大量逐 kernel 文本输出带来约 0.35 s 的可见扰动。

Reactive 每层宽度为 6 或 12，而 GPU 数为 2，因此当前 `WIDE_DAG_GUARD` 会拒绝没有“实测吞吐超线性收益”的 2-way Split。对这组运行，真正被直接验证的是 P3 的 admission 结果。P1 canonical merge 没有实际 merge 可以执行，P2 single-first/hysteresis 也没有进入一次 Split probe；不能用这份日志声称 P1/P2 的运行数据路径已经完成性能消融。

### whole-kernel placement 的实际形态

2400 个 single kernel 的设备分配为：

```text
device 1 = 1949 kernels
device 2 =  451 kernels
```

冷启动后的稳态基本固定为：device 2 承担 member 3 的 `R/X/Y/F` 四个 kernel，device 1 承担其余五个 member 的 20 个 kernel。按本程序已知的 RAW 边统计，全部 3594 条同窗口及跨窗口相邻依赖中只有 42 条跨设备，约 `1.17%`；step 5 以后除 step 11 的一条外均保持 member-local。由此可以得出：

- HEFT 已经学会了 producer-consumer affinity；
- Reactive 上 P4 所担心的“每个窗口随机迁移完整 member”没有从 placement 日志中出现；
- 20:4 的 kernel 数不等于 5:1 的工作量失衡，必须看每卡实测 service time。

最后一个 window 的预测单设备成本求和约为：

```text
device 1 assigned cost = 7080.247 × 10 us = 70.80 ms
device 2 assigned cost = 10235.19 × 10 us = 102.35 ms
predicted window lower bound                    = 102.35 ms
```

100 个 window 仅该下界就约为 `10.24 s`，已经接近原生整次 `10.154259 s`。日志中 device 2 上 member 3 的稳态 `R/X/Y/F` 预测分别约为 `35.4/15.0/15.0/37.0 ms`；而 device 1 上其他 member 的同类 kernel 普遍低得多。`single_exec_cost` 包含 event profile EWMA、跨设备 capability scaling 和 `monitorPenalty`，所以它还不能区分以下两种情况：

1. 第二张 4090 在当前机器/进程中确实具有更低的有效 service rate；
2. profile、瞬时 monitor penalty、event 归属或设备队列状态把第二张卡估慢了。

两张卡名义上都是 RTX 4090，因此在逐卡隔离基线和 raw profile/penalty 分列完成前，不能把差异直接解释成硬件异构，也不能用强制 50:50 kernel 数量来“修正”它。

### 当前 11 秒平台的正确归因

这轮结果把问题从 Split 数据路径推进到了 whole-kernel 调度层：

- 已确认：旧 92 s 的主因是错误持续 Split；
- 已确认：默认 guard 已把 Split input/merge 路径完全移出本次执行；
- 已确认：稳态依赖局部性良好，placement 没有频繁抖动；
- 已确认：双卡有效成本矩阵高度不对称，预测关键路径落在 device 2；
- 尚未确认：device 2 的 raw kernel time、`monitorPenalty`、普通 D2D bytes、每卡 active overlap 和离线控制面各占多少；
- 尚未确认：offline 强制只用最佳 GPU 的时间，因此还不能区分“离线运行时固定开销”与“第二 GPU 的净收益”。

因此新结论不是“逻辑并行仍不足”，而是：逻辑 width 已经足够，Split admission 也正确；但当前任务粒度、实测设备速度和运行时固定成本组合后，没有形成超过最佳单卡的净吞吐收益。

## 哪些修正只是边界，哪些真正影响性能

| 修正 | 性质 | 性能影响 |
|---|---|---|
| Split device 去重、范围检查、part 数整除 | 边界/正确性 | host 判断成本极小；非法候选退回单 GPU |
| 按 AccessRange、offset 和 owned row block merge | 正确性 | 通常减少错误/多余 copy；线性化本身成本很小 |
| sub-buffer、非连续 trailing range 拒绝 Split | 能力边界 | 可能失去某些 Split 机会，但当前实现无法安全执行 |
| read_write 完整 snapshot + owned write merge | 核心语义 | full-read replication 和 merge 都是实际成本 |
| CG accessor/scalar 深复制 | 正确性 | 只增加少量 host 参数复制，可忽略 |
| GlobalOffset 保留原 offset 后再加 part offset | 正确性 | 无实质性能影响 |
| context+device allocation lookup | 正确性 | 查找成本很小，避免错误 pointer |
| 每设备 private context | 正确性 + 数据路径 | 使跨设备 copy 依赖 peer capability；不可用时 staged copy 很贵 |
| CUDA context/range validation | 边界/诊断 | 几个条件判断，通常可忽略 |
| direct peer D2D | 性能路径 | 成功时避免 host staging，是重要优化 |
| D2H→H2D fallback 且逐 copy wait | 正确性 fallback | 当前最显著的数据移动开销之一 |
| pending event + consumer-driven deferred merge | 原核心性能设计 | 保留独立 kernel overlap，避免过早 merge |
| 记录 pending read/write set | 正确性 | 写 hazard 可能更早 finalize，但这是必要依赖 |
| non-split read replica reuse | 安全优化 | 减少一次不必要移动 |
| 禁止未经版本证明的 split-to-split replica reuse | 正确性收紧 | 增加输入复制，性能影响明显 |
| 每设备稳定 in_order queue | 模型一致性 + 正确性 | 关闭同 GPU 隐式多 stream overlap，影响明显；跨 GPU 并行保留 |
| 相关 pending 先全部 wait、再 merge | 正确性/归因 | consumer 通常本就要等全部输入；batch end 可能减少 merge/compute overlap |
| multipart mqueue | 控制面边界 | 大 DAG 多几个消息 chunk；避免 8192-byte 截断/死锁 |
| split wall-time profile | 成本模型正确性 | profile 包含真实 copy/merge；计时本身开销很小，但会改变后续调度决策 |
| 前驱 read bytes 从 Split copy cost 中扣除 | 当前成本模型缺口 | 没有完整 per-device version 证明时会严重低估 X/Y/F 的 Split 成本 |
| single 与 Split profile 按 part 数隔离 | profile 语义正确但缺少探索 | 避免混淆不同执行方式；若没有 single baseline，会让首次 Split 决策长期自我强化 |
| 宽 DAG 中逐任务贪心 Split | 调度策略缺口 | 一个 Split 同时占用两台 GPU，可能损失更有价值的 member/task parallelism |
| per-part 日志 | 诊断 | 开启大量 trace 时会影响性能，正式性能实验应关闭详细输出 |

真正需要在性能结论中单独量化的是：

1. full read replication；
2. peer D2D 成功率；
3. staged D2H→H2D bytes 和等待时间；
4. split output owned-row merge；
5. stable in_order queue 带来的同 GPU overlap 变化；
6. split-to-split replica 不复用造成的额外复制；
7. profile/daemon 的每窗口固定成本。

不能把这些成本统称为“Split 开销”，否则无法判断应该优化 Layer B 的调度决策，还是 Layer C 的数据版本和移动实现。

## 修正后仍然保留的异步能力

当前设计仍然允许：

- 同一 Split kernel 的不同 part 在不同 GPU 并行；
- 不同 member 的独立 pipeline 放到不同 GPU；
- X/Y 或不同 member 的无依赖 kernel 在不同 GPU overlap；
- split merge 延迟到真实 consumer；
- 不触及 pending read/write set 的 kernel 继续提交；
- batch 内按 HEFT 顺序执行，而不是源码提交顺序；
- 下一物理时间步使用前一窗口 profile 修正 cost。

当前没有保留的是：

- 同一 GPU 上不受模型约束的多 out-of-order queue overlap；
- 没有 version epoch 证明的 split replica shortcut；
- 在 peer 不可用时把跨 context pointer 强行送入普通 D2D；
- 让后一个未确认 kernel 排在前一个 merge 前并把错误归错对象。

这些被移除的行为不是系统设计要展示的异步性，而是尚未被调度模型或数据一致性模型表达的隐式并发。

## 与 Celerity 三层缺陷的对应关系

### Layer A：访问区域不等于计算成本

R、X/Y、F 的 mapper 都可以准确描述数据范围，但 mapper 不能表达：

- R/F 中 chemistry substeps；
- member 间网格和反应成本差异；
- 不同 GPU 对 chemistry 和 WENO 的相对速度；
- profile 后发现某个 Split factor 是否真正更快。

Reactive 用真实 stage/member 异构性支持独立 cost model 和 EWMA profile 的动机。

### Layer B：每 task 确定性切分不等于全 DAG 调度

一个时间步同时包含：

- 多个完全独立 member；
- 每个 member 内独立 X/Y 分支；
- 可能值得 Split 的大 R/F；
- 不值得 Split 的粗网格 member；
- producer-consumer locality。

系统要联合选择整 kernel 放置、Split degree、device subset 和执行顺序，而不是让每个 task 固定参与全部设备。

### Layer C：细粒度 coherence 与完整 replica/merge 的不同代价

Celerity 的 neighborhood mapper 可以只为 WENO 交换 radius-3 halo，virtual buffer 也不要求每设备完整 backing。当前系统则为 Split read 准备完整 replica，再把 owned output rows merge 到一个完整版本。

因此 Reactive 不是假设当前系统在单个 WENO 数据维护上天然更强。它的正面假设是：通过 Layer A/B 的 profile-guided whole-member placement 和 selective Split，避免让所有 task 都付出分布式 coherence/participation 成本。

如果 full replication 和 staged copy 最终超过调度收益，这应被诚实归因到 Layer C，而不能说 Celerity 的 mapper 有问题。

## 当前边界和下一步性能验证

当前 Split 仍主要支持：

- NDRange dim0；
- 完整 trailing dimensions；
- row-major contiguous row block；
- partition-disjoint writes；
- merge 后形成一个完整 current version。

它还没有：

- 通用 halo-aware read preparation；
- per-device version/replica epoch table；
- 无 host wait 的 copy→kernel event chain；
- 在调度模型中显式表示 compute/transfer 多 stream；
- column/block-aware merge；
- 从普通 `read_write` accessor 自动推导真实写集合。

`8192×8192, 100 steps` 已经完成并验证正确。原计划中的 split-disabled 对照、single baseline、wide-DAG guard 和 selected-Split 计数已经完成；Reactive 默认运行已从 2400 次 Split 变为 2400 次 single。下一轮不应继续修改 Split merge，而应按以下顺序隔离 11 秒平台：

1. 分别在两张物理 GPU 上运行相同的原生单卡基线，记录 device event time、时钟、功耗、温度和外部占用，确认第二张卡的有效速度差异是否真实；
2. 增加 offline “强制 device 1”与“强制 device 2”的 whole-kernel 对照，和当前 split-disabled 双卡 HEFT 分开，量出离线控制面的固定成本及第二张卡的边际收益；
3. 把 raw event EWMA、capability scaling、`monitorPenalty` 和最终 candidate cost 分列，不再只输出合成后的 `single_exec_cost`；
4. 补齐普通 non-Split 的 H2D/D2D/D2H bytes、每卡 kernel active time、batch predicted makespan 与实际 window wall time；
5. 为“多设备 whole-kernel schedule”相对“最佳整批 co-located schedule”增加 batch-level gain margin，预测收益不足以覆盖控制面和迁移成本时回退最佳单卡；
6. 若逐卡原生速度接近但 raw offline profile 仍相差数倍，再检查 queue/event/device 归属、monitor 采样陈旧和 profile warm-up/outlier，而不是强制平均分任务；
7. 用窄 DAG 或 `single-large` 控制模式单独验证 P1/P2 selective Split；Reactive 宽 DAG 的 `split=0` 本身是正确结果；
8. 关闭详细 trace 后，对比 native 两张单卡、offline 最佳单卡、双卡 whole-kernel、selective Split 和 Celerity。

如果正确性通过，性能分析必须同时报告：

- `run_sec` 与最终 host materialization 时间；
- raw event profile、monitor penalty 和最终预测成本；
- placement 和 Split degree；
- Split 与普通 whole-kernel 的 H2D、D2D、D2H bytes/time；
- peer fallback 次数；
- 每 GPU active time、overlap、时钟和利用率；
- 最佳 co-located 与 multi-GPU predicted/actual window time；
- cold window 与 steady-state window。

## 结论

Reactive Transport Ensemble 不是为了迁就 runtime 而把真实算法削成简单 element-wise data parallel。它保留了刚性反应、高阶二维 stencil、真实 fan-out/fan-in、多 fidelity ensemble 和重复时间步。

持续修正的方向也不是逐步取消异步 Split，而是把原来隐含、粗糙的假设显式化：

- 什么写布局可以 Split；
- read_write 需要什么 snapshot/merge 合同；
- 每个 clone 拥有哪些参数；
- 一个 allocation 属于哪个 context/device；
- 哪个 replica 对应哪个 version；
- 何时 consumer 真正需要 merge；
- HEFT 的单设备时间线如何映射到 runtime queue；
- CUDA 700 是在哪个异步 kernel 产生，而不是在哪个 API 被观察。

修正后的系统比原型更保守，但仍保留跨 GPU 的 task parallelism、intra-kernel Split 和 dependency-driven deferred merge。最新 2400-kernel 长运行证明 Reactive 已经从 CUDA 700 排错案例变成了稳定的正确性与调度压力案例；它同时否定了“只要 kernel 大、DAG 宽就一定能从 Split 获益”的简单判断。第一轮 admission 修正是有效的：`92.194035 s -> 11.459888 s`，并且 checksum 不变。

在当前数据语义下，Reactive 的调度已经把独立 member 长时间固定在不同 GPU 上，并在宽 DAG 中正确选择了零 Split。剩余问题不再是“如何继续禁止 Split”，而是为什么两张名义相同的 4090 形成高度不对称的有效成本，以及双卡 predicted gain 为什么没有覆盖约 13% 的系统开销。逐卡校准、普通数据移动计数和 batch-level no-regression admission 完成前，Reactive 仍不应被用作系统优于 native 或 Celerity 的正面性能结果。
