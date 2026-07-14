# Multi-replica WCSPH miniapp 说明

对应源码：`sph_ensemble_sycl_timer.cpp`。

## 1. 五类 kernel 与物理含义

每个物理时间步按全 replica 分层提交，而不是逐 replica 提交完整链：

1. `Density`：用固定 stride Verlet 邻居行和 Wendland C2 核计算密度。
2. `EOS`：用 `gamma=7` 的 Tait 状态方程由密度计算压力。
3. `PressureForce`：遍历真实邻居，计算对称压力梯度加速度。
4. `ViscosityForce`：遍历真实邻居，用平滑 Laplacian 计算黏性加速度；它不读 pressure，因此可与 `EOS -> PressureForce` 分支并行。
5. `Integrate`：合并两个加速度分支和重力，更新并周期 wrap 位置/速度。

计算量没有 `repeat` 或 busy loop；三个重 kernel 的工作量均来自 SPH 邻居相互作用。邻居表在 host 初始化阶段通过周期 cell list 构造，条目都满足真实 Verlet cutoff，初始化不计入 `run_sec`。

## 2. Buffer 读写合同

| Kernel | 读取 | 写入（每个 work-item `i` 只写元素 `i`） |
|---|---|---|
| Density | `position[cur]` 全域、owned `neighbor_count/index` 行 | `density[i]` discard-write |
| EOS | `density[i]` | `pressure[i]` discard-write |
| PressureForce | `position[cur]`、`density`、`pressure` 全域，owned 邻居行 | `pressure_accel[i]` discard-write |
| ViscosityForce | `position[cur]`、`velocity[cur]`、`density` 全域，owned 邻居行 | `viscosity_accel[i]` discard-write |
| Integrate | `position[cur][i]`、`velocity[cur][i]`、两个 acceleration `[i]` | `position[next][i]`、`velocity[next][i]` discard-write |

每个 replica 有独立的 10 个 SYCL buffer identity：双缓冲 position、双缓冲 velocity、density、pressure、两个 acceleration、neighbor count 和 neighbor index。没有把 replica 拼成一个大 NDRange。

## 3. DAG、宽度与关键路径

单 replica DAG：

```text
position[t], velocity[t]
          |
          v
      Density[t]
       /       \
      v         v
   EOS[t]   ViscosityForce[t]
      |         |
      v         |
PressureForce[t]|
       \        /
        v      v
       Integrate[t]
            |
            v
 position[t+1], velocity[t+1]
```

对 `R` 个 replica：每步 `5R` 个 kernel，最大结构宽度 `2R`，关键路径 4 层，等成本结构平均并行度为 `1.25R`。默认 `R=64` 时分别是 320、128、4 和 80。默认 `window_steps=1`，只在物理步末 `queue.wait()`；增大该参数可以把多步连接成一个压力测试窗口。

## 4. 为什么主模式应 whole-replica placement

`mixed` 默认已有 64 条互相独立的 replica pipeline，两张 GPU 在最宽层面对 128 个 ready task。继续把单个 kernel Split 不会增加有用设备并行度，却会为 indirect gather 建立完整 read snapshot、merge owned writes，并破坏同一 replica 的 producer affinity。合理策略是将不同 whole replica 分给不同设备，并让其后继沿用已有 position、velocity、neighbor table 和临时场的位置；默认应选择零或极少 Split。

`single-large` 刻意只保留一个 replica，是低 task-parallelism 的边界控制。这里 Split/Celerity 分布可能更合适，结果应独立报告。

## 5. Split 为何正确但昂贵

所有 NDRange 都是一维 `N`，work-item `i` 只写输出元素 `i`。程序要求每个 class 的 `N % split_parts == 0`，所以 dim0 part 的 owned write 是连续、互不重叠的；只 merge owned range 语义正确。

但 neighbor index 是 buffer 内容中的动态间接索引。运行时只从 accessor 元数据无法知道某个 execution chunk 读取哪些 position/density/pressure 元素，因此安全 Split 必须给参与设备完整 read snapshot。它是 `O(N)` 可见性维护，不是理想 halo；benchmark 没有假装 runtime 能推导不存在的 footprint。

## 6. Celerity 的准确 mapper

公平 Celerity 版本应使用：

| Kernel | Mapper |
|---|---|
| Density | execution 为粒子区间；count `one_to_one`；index 自定义 `[b,e) -> [b*Kmax,e*Kmax)`；position `all`；density 输出 `one_to_one` |
| EOS | density/pressure 都是 `one_to_one` |
| PressureForce | 邻居行 slice；position/density/pressure `all`；acceleration `one_to_one` |
| ViscosityForce | 邻居行 slice；position/velocity/density `all`；acceleration `one_to_one` |
| Integrate | 所有输入输出均 `one_to_one` |

这里 force/density 的 `all` 是 indirect gather 的语义上界，不是故意使用差 mapper。若所用 Celerity 版本支持 task hints、worker subset 或把 replica 固定到单 worker，必须启用并把更快结果作为公平对手。

## 7. 三层假设对应的测量

- Layer A（计算成本）：记录 class/device/part 的 raw profile、EWMA、预测误差，以及 Density、PressureForce、ViscosityForce 的 native 分位数。class 至少由不同 `N`/buffer shape 区分，未用隐藏 repeat 制造不可学习成本。
- Layer B（调度）：记录每设备 assigned predicted/actual work、active overlap、window makespan、placement 稳定性、Split 数及 no-Split 对照。
- Layer C（数据维护）：记录每步 H2D/D2H/D2D bytes、完整版本所在设备、step 5 后的迁移，以及 Celerity copy/push/await/collective bytes 和 command 数。

## 8. 已验证事实与待验证假设

已完成的代码级验证：

- C++17 语法通过本地轻量 SYCL API stub 检查。
- host 顺序执行 stub 下跑过 `uniform`、`mixed`、`single-large` 小规模案例。
- AddressSanitizer 和 UndefinedBehaviorSanitizer 下，小规模 mixed/reference 案例无内存或 UB 报告，`VERIFY passed=1`。
- 小规模 reference 会对每个 class 的一个代表 replica、`N<=2048` 做全量 position/velocity/density/pressure/acceleration 比较；正式规模对全部最终 position 做位移和周期边界检查。

尚未实测、不能当作现有结果的假设：真实 SYCL 编译器/设备运行、三个重 kernel 是否达到 3 ms 门槛、双 GPU overlap/speedup、steady-state 迁移量、Split 选择，以及相对最佳公平 Celerity 是否达到 1.20x。当前机器只有 SYCL 源码头且缺 OpenCL headers/可用 SYCL compiler，因此没有伪造硬件性能数字。

## 9. 编译与运行

通用 DPC++/Intel oneAPI 示例：

```bash
cd test-scenario
icpx -O3 -std=c++17 -fsycl sph_ensemble_sycl_timer.cpp -o sph_ensemble_sycl
```

小规模 reference/smoke test：

```bash
./sph_ensemble_sycl \
  --particles 2048 --replicas 4 --steps 3 \
  --max-neighbors 192 --classes 2 --coarsen-percent 10 \
  --mode mixed --verify 1 --host-read-full 1
```

默认主候选（约束和精确内存量由程序启动时打印）：

```bash
./sph_ensemble_sycl --mode mixed --memory-limit-gib 16 --verify 1
```

控制模式：

```bash
./sph_ensemble_sycl --mode uniform --verify 1
./sph_ensemble_sycl \
  --mode single-large --particles 1048576 --replicas 1 \
  --max-neighbors 192 --verify 1
```

`--wait-each-kernel 1` 只用于排错，不能用于正式结果。正式跑前应先用 native profile 调整真实 `particles/max-neighbors`，使三个邻居 kernel 的第 25 百分位达到设计门槛，同时保证 `max_displacement < skin/2`。

## 10. 公平实验表

每组至少 5 次，关闭详细 trace，报告 median 和 min；所有系统必须使用相同物理参数、双 GPU 和正确性容差。

| 编号 | 配置 | 用途 |
|---|---|---|
| A0 | native SYCL，物理 GPU 0 | 最佳单卡候选 |
| A1 | native SYCL，物理 GPU 1 | 设备校准 |
| B1 | offline 强制 device 1、no Split | 控制面成本 |
| B2 | offline 强制 device 2、no Split | 第二设备/queue 路径 |
| C | offline 双 GPU、强制 no Split | whole-replica placement 主对照 |
| D | offline 双 GPU、默认 selective Split | 检查是否正确拒绝无收益 Split |
| E | Celerity、准确 mapper、默认策略 | 默认 visibility/distribution 成本 |
| F | Celerity、版本所支持的最佳 hints | 最强公平对手；E/F 取更快者 |
| G | `single-large` 的 native/offline/Celerity | Celerity-friendly 边界控制 |

主结论只有在正确性、kernel 粒度、双卡 overlap、step 5 后驻留、no-Split 对照和 `speedup_over_celerity >= 1.20` 等设计门槛全部满足后才能成立；否则该程序只能作为调度测试候选，不能宣称已经证明系统优于 Celerity。
