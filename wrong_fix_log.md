已完成分析并修正 benchmark。结论是：单设备结果整体符合结构预期；调度系统卡住的直接原因是 queue.wait() 没有进入离线调度入口，而不是第 306 个 kernel 本身出错。
结果判断
[base-result.txt (line 2)](/Users/heyifu/Work/SYCL/test-scenario/base-result.txt:2) 的三组结果结构正常：
内存估算与参数匹配，均输出了完整 TIMING/RESULT，结果有限。
2048² → 16384² 数据量增大 64 倍，而运行时间约增大 12.4 倍，符合小规模受 kernel 启动/JIT 开销影响、大规模提高 GPU 利用率的可能表现。
cycles 从 2 增至 10，kernel 数从 306 增至 1546，运行时间从 13.04 秒增至 51.26 秒，趋势合理；首次 JIT 等固定开销会令增长小于 5 倍。
但 verify=0 且只读取稀疏 sample，因此只能证明程序完成并产生有限结果，不是严格数值正确性证明。正式性能数据应多次运行取中位数。
卡住原因
[our-result.txt (line 46)](/Users/heyifu/Work/SYCL/test-scenario/our-result.txt:46) 恰好记录了 306 个 deferred CG：
每个 Arnoldi cycle = Σ(k=0..7)(8 + 3k) = 148
总数 = 2*148 + 4(marker/projection) + 5(restart) + 1(sample) = 306
之后没有 mq_send kernel_req_datas、HEFT、resubmit、TIMING 或 RESULT，说明 CG 注册完成，但调度分发尚未开始。
直接原因：
[event.cpp (line 78)](/Users/heyifu/Work/SYCL/code-llvm-sycl/sycl/source/event.cpp:78) 只有公共 event::wait() 会调用 scheduleOffline()。
[queue_impl.cpp (line 463)](/Users/heyifu/Work/SYCL/code-llvm-sycl/sycl/source/detail/queue_impl.cpp:463) 的 queue.wait() 只执行内部 event wait/piQueueFinish，不会进入上述入口。
原程序最终使用 queue.wait()，因此延迟保存的 306 个 CG 没有被提交。
还有一个紧随其后的潜在死锁：运行时消息上限是 8192 字节，[handler.cpp (line 2708)](/Users/heyifu/Work/SYCL/code-llvm-sycl/sycl/source/handler.cpp:2708) 会把整个窗口序列化为一条消息，并且没有检查 mq_send 返回值。306 个 kernel 必然超限，随后会阻塞在 mq_receive。
日志中的 No protocol specified 和 GPU 1–4 映射不是此次卡住的原因，因为 daemon 后续已经正常建立并注册程序。
已完成的修正
[krylov_gmres_sycl_timer_wrong_fix.cpp (line 52)](/Users/heyifu/Work/SYCL/test-scenario/krylov_gmres_sycl_timer_wrong_fix.cpp:52) 现在：
保存每次 queue.submit() 返回的 event，以 event::wait() 刷新调度窗口。
默认 --window-kernels 24；最坏序列化估算约 7389 字节，低于 8192 字节，同时仍保留多 kernel DAG。
不再默认调用 queue.wait() 或 wait_and_throw()。
--wait-each-kernel 1 使用逐 event wait，仅用于调试。
根据当前运行时的整 buffer 依赖粒度，将 Q/H 访问改为整 buffer accessor，并在 kernel 内显式使用 offset。
Q 的局部写入 kernel 增加一元素 write guard，防止 SNMD 按错误的 (m+1)N 范围拆分/合并。
operator-only 只包含重复 stencil，不再为了统一输出额外执行整向量 copy。
所有 buffer 使用持久化 std::vector backing，保证跨 rank 路径调用 getUserPtr() 时有有效 host pointer。
增加单 buffer 2 GiB 安全检查及 RUNTIME compatible=... 输出。旧的 16384²,m=8 Q 为 9.66 GB，原生 SYCL 可以运行，但当前调度协议的 int 大小字段无法安全表示。
host/device 初始化现在使用相同的 q0 归一化。
分析和运行约束也已记录在 [krylov_gmres_sycl_notes_wrong_fix.md (line 37)](/Users/heyifu/Work/SYCL/test-scenario/krylov_gmres_sycl_notes_wrong_fix.md:37)。
按要求没有在本工作区构建；已完成静态审计和 git diff --check。由于窗口同步、host backing 和大 buffer guard 均有变化，旧 native 时间不能直接用于最终 speedup，建议用修改后的同一源码重新测 native 与调度版本。