# Covariance/Subspace bench 调度分析与修复

## 结论

这个 workload 的纯 DAG 并行宽度是 8，但现有 kernel 形状没有给四块 GPU
提供足够的设备内并行度。原生单卡依靠 out-of-order queue，把 8 个 band 的同层
kernel 同时放到一块 GPU 上，用 kernel 间并发填充 occupancy。当前 SNMD 路径则把
每块 GPU 当成独占资源，并强制每块设备使用 in-order queue，因此每层只能同时运行
4 个 kernel，必须执行两波；同时 band 在设备间漂移，产生额外数据迁移和同步。

对 `covariance-result.txt` 中三个 wait window 的 event profile 聚合后：

| window | 实际 epoch | 同层完全重叠下界 | 四卡、每卡一个 kernel 的阶段下界 | 调度器记录的迁移量 |
|---:|---:|---:|---:|---:|
| 1 | 7.702 s | 1.841 s | 3.682 s | 352.1 MiB |
| 2 | 5.949 s | 1.805 s | 3.609 s | 192.0 MiB |
| 3 | 6.278 s | 1.830 s | 3.659 s | 304.0 MiB |

原生单卡稳态约 2.09 s，已经非常接近 1.80--1.84 s 的同层重叠下界。因此，在不改变
kernel 实现的前提下，多设备调度没有可利用的剩余关键路径并行度；最优策略是保留
一块 GPU 上的 kernel 间并发，而不是把同层的 8 个低 occupancy kernel 分散到四块
GPU。

## Bench DAG

每个 epoch 有 8 个完全独立的 band。每个 band 是一条 19 层的链：

1. mean；
2. stddev；
3. normalize；
4. triangular correlation；
5. 8 个 reverse GS panel；
6. 前 7 个 panel 各有一个 update，最后一个 panel 没有 update。

所以一个 wait window 有 152 个 kernel，宽度 8、关键路径 19。三个 epoch 共提交
456 个 kernel。

最重的 correlation kernel 只有 2048 个 work-item。每个 work-item 串行计算一个三角
矩阵行，内部循环长度很大，总计约 8.594G FMA/band。这是“大工作量、低设备并行度”
而不是能单独填满 GPU 的大 kernel。日志中的单个 correlation event 约 1.42--1.45 s；
原生单卡让 8 个这样的 kernel 同层并发，整层仍接近一个 event 的延迟。

## 修复前的完整路径

1. `handler::finalize()` 不立即向 backend 提交 kernel，而是把 command group、原 queue、
   kernel identity、ND-range 和每个 requirement 的访问模式/范围记录到
   `ProgramManager::kernel_cgs` 与 `kernel_reqs`。
2. 用户的 `queue.wait_and_throw()` 最终进入 `queue_impl::wait()`，由一个 flush handler
   调用 `handler::scheduleOffline()`。
3. handler 为这个 wait window 编号，把 152 个 `S2DKernelReqData` 通过 POSIX message
   queue 发给 daemon。
4. daemon 将 requirement 按 buffer identity 和访问模式构造成 RAW/WAR/WAW DAG。
   本 bench 得到 8 条互不相交的 19 层链。
5. `algorithmHEFT()` 注册 kernel feature，读取 persisted/live profile、monitor 负载、
   通信带宽和内存容量，计算 upward rank，再用统一 risk objective 比较每个 Single 和
   Split placement。
6. correlation 的 Split 在当前实现中被拒绝：一维 ND-range 写二维完整 accessor 会触发
   `dim0-only kernel writes non-contiguous range 2048x2048x1`。实际执行的 456 个 kernel
   全部是 `NumParts=1`。
7. batch HEFT 完成后，单 rank 路径进入 completion-driven queue。该路径清空静态
   placement 并动态重选 ready node；选中一个 node 后，把其 GPU calendar 设置为
   infinity，直到 handler 回报 event complete。因此每块 GPU 同时最多有一个 kernel。
8. daemon 一开始发出最多 4 个 ready kernel。handler 为每块设备创建一个私有 context
   和稳定的 in-order profiling queue，必要时迁移 buffer，然后 `resubmit()`。
9. handler 轮询 in-flight event；完成一批就把 duration 和 placement 回传 daemon。
   daemon 释放对应 GPU，重新选择下一批 ready node。这个往返持续到 152 个 kernel
   全部完成。
10. 用户 wait fence 返回；下一个 epoch 重新建立一个 152-kernel window。历史 DAG 仍被
    用作 residency anchor，但计算 calendar 被 rebase。

这里有两个相互强化的问题：

- completion-driven queue 的 infinity reservation 禁止同一 GPU 上的独立 kernel 并发；
- handler 的稳定 profiling queue 被强制设为 in-order，即使一次发送多个 kernel 也无法
  恢复原生 queue 的并发行为。

此外，HEFT 会为了平衡独占时间线而让同一 band 的相邻阶段换设备。日志中的稳态窗口仍
记录了 192--304 MiB 的迁移；部分 `resubmit()` 会在准备跨 context 数据时同步阻塞，进一步
扩大 3.6 s 阶段下界与约 6 s 实际时间之间的差距。

## 理论上能否快于单卡

如果只看 DAG 并假设每个 kernel 在任何 occupancy 下都有固定吞吐，8 个 band 在四块设备
上具有最高 4 倍的 task-parallel 上界。但这个假设与实测不符：单独运行的 2048-work-item
correlation 已是约 1.43 s，而原生单卡通过同时驻留 8 个 band，把整层也压到大约这一延迟。

当前 kernel 的调度下界是各层最大 event 延迟之和，即约 1.8 s。把同层 kernel 移到更多
设备不能把某个 event 本身缩短；只会把一块 GPU 上已有的 latency hiding 分散掉。要获得
真正的多卡加速，需要先改算法，例如把 correlation 的 snapshot reduction 改成二维并行
加分层归约，使一个 band 本身能有效占用或拆分到多块 GPU。这已经不是调度器能够透明完成
的变换。

打印的 `deterministic_split_speedup_ceiling=2.286` 也不是当前程序的可达加速：它只根据
前向三角行等长切分后的最大工作份额 43.74% 计算，未计入每个 part 仅剩 512 work-item 后
的 occupancy 损失、输入复制、输出 merge，也未计入当前 Split 合法性检查会直接拒绝该
kernel。

因此对现有二进制，系统的正确目标不是强制使用四卡，而是自动回退到接近原生单卡的
co-located 并发路径。

## 实现的修复

### daemon

`applyCoLocatedGpuScheduleIfBetter()` 现在为同一 GPU 的 out-of-order 批次建立 occupancy
感知模型。对同一 DAG depth 中的 kernel `i`：

```
demand_i = min(1, global_work_items_i / target_items)
level_cost = max(max_i(exec_i), sum_i(exec_i * demand_i))
```

默认 `target_items=16384`，约等于 Ada GPU 上每个 SM 四个 resident warp 的总 work-item
数量。满载 kernel 的 demand 为 1，仍按工作量相加，因此不会因为“同层并行”就被错误地
收缩到单卡；低 occupancy kernel 可以共享一个 residency budget。相同计算也应用于
profile uncertainty 的 risk 上界，最后与普通多设备 HEFT 的 risk 上界比较。

可以用 `SYCL_SNMD_CONCURRENT_TARGET_ITEMS` 校准阈值；设为 `0` 会禁用该模型。

若 co-located 候选获胜，daemon 保留该静态 placement，并跳过 completion-driven queue，
避免后者再次清空 placement 并恢复独占设备语义。

### handler

handler 现在为每个设备分别缓存两种 profiling queue：

- 普通 HEFT/Split 路径继续使用 in-order queue；
- daemon 选择的单设备静态批次使用 profiling-enabled、非 in-order queue。

所有 152 个 command group 按拓扑序提交到同一私有 context。SYCL buffer dependency 仍会
串行化一个 band 内的 19 个阶段；不同 band 使用不同 buffer，因此 backend 可以像原生
运行时一样并发执行同层 kernel。

## 目标端验证

重新编译 daemon/runtime 后，用原命令运行即可。修复命中时应看到：

```
algorithmHEFT: co-located GPU batch schedule selected ... concurrent_target_items 16384
=== handler === Offline static concurrent batch: 1
=== handler === Offline profiling queue created, profiling: 1 in_order: 0
```

该 window 不应再出现 `CompletionQueue: dispatch_order ...`。所有 `num_parts` 应保持 1，
所有 kernel 应落在同一个 device；第二、第三个 epoch 不应再有跨 GPU 的 buffer 迁移。

性能目标是稳态 epoch 接近原生单卡的约 2.09 s，而不是理论上不可达的四卡线性加速。
验证仍应输出相同 checksum，并保持 `VERIFY passed=1`。

