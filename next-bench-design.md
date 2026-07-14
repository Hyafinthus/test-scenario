# 下一代正面 Miniapp 唯一设计规格：Multi-replica WCSPH Ensemble

## 本文档的用途

本文档应当能够作为新对话中的唯一背景，直接指导实现下一代 miniapp。实现者不应再依赖 Reactive、Krylov 或此前聊天记录来补全关键设计。

目标源码文件：

```text
test-scenario/sph_ensemble_sycl_timer.cpp
```

最终任务是构造一个真实、可验证、逻辑并行度极大、重 kernel 占主要运行时间，并且符合当前运行时边界的科学计算 miniapp。主性能模式必须依靠大量独立但内部有真实多阶段依赖的 SPH replica pipeline 获益，不能依靠持续 Split，也不能退化成一个简单逐元素 kernel 的批量复制。

实现风格遵守 `3mm_timer.cpp` 和现有 Reactive timer：单个 C++17 源文件、SYCL buffer/accessor、清晰区分初始化、queue、run、host materialization 和 total 时间，输出稳定的 `CONFIG/DAG/MEMORY/TIMING/RESULT/VERIFY` 行。

## 最终设计决策

放弃把规则 3D stencil/wave propagation 作为当前系统的主正面应用，改用：

> 多副本弱可压缩光滑粒子流体动力学（Multi-replica Weakly Compressible Smoothed Particle Hydrodynamics，WCSPH）ensemble。

每个 replica 是一个私有粒子系统，执行密度、状态方程、压力力、黏性力和积分组成的重复物理时间步。不同 replica 可以表示不同粒子数、平滑长度、黏度、声速和邻居密度等级，是参数扫描、不确定性量化和 ensemble simulation 中真实存在的并行单位。

选择它的原因是：

- 默认 64–96 个 replica，可以长期提供 64–192 的结构宽度，而双 GPU 只有两个执行资源；
- 每个 replica 内有真实 fan-out/fan-in 和跨时间步状态链，不是独立单 kernel；
- 三个主要 kernel 都包含大规模邻居遍历，计算量来自 SPH 物理公式；
- 每个 replica 的位置、速度、邻居表和临时场完全私有，whole-replica placement 后可以长期驻留；
- 间接邻居 gather 对 Celerity 必须使用语义正确的保守全域 read mapping，而不是故意使用错误 mapper；
- 当前系统可以通过 profile-guided whole-kernel placement 避免把每个 replica 都分布到两张卡；
- 所有 work-item 只写自己的粒子元素，dim0 Split 的 owned write 区间连续且互不重叠；
- 默认宽 DAG 应选择零 Split，`single-large` 模式则保留为 Celerity 友好的分布式控制项。

### 分模式的胜负预期

这里的正面结论必须明确：**主 `mixed` 模式预期由本系统胜过 Celerity，Celerity 更慢。** 这不是要求本系统在所有形态下都胜过 Celerity，而是一个有主次之分、必须由实测验证的研究假设：

| 模式 | 预期与用途 |
|---|---|
| `mixed` | 主正面模式。本系统必须胜过同硬件、准确 mapper、启用其版本所支持最佳 hints 的 Celerity；否则此 miniapp 不能作为“本系统优于 Celerity”的正面结果。 |
| `uniform` | 机制控制。本系统应通过大量独立 whole-replica pipeline 保持良好双卡吞吐，但不单独承担最终胜负结论。 |
| `single-large` | Celerity-friendly 反例/边界控制。这里只有一个可分布的大 replica，Celerity 可能更快；该结果用于证明我们没有把所有 workload 都包装成有利于自己的形态。 |

`mixed` 中预期胜出的原因不是 Celerity mapper 写错，而是：本系统从整个 wait-window DAG、实测 kernel 成本、设备速度、producer affinity 和已有数据位置联合决定 whole-replica placement；Celerity 即使能用 hint 避免单个 replica 的无谓分布，其固定或用户指定 placement 仍缺少同等的动态加权全局决策。若所测 Celerity 版本已经能自动得到同等 placement、驻留和调度成本，并在 `mixed` 中追平或胜出，就说明这一正面假设不成立，必须换设计或继续改 runtime，不能把它解释成本系统占优。

## 当前运行时合同

实现者必须以以下能力为边界，不得假设尚不存在的功能。

### wait-window DAG

- `SCHEDULE_OFFLINE` 在 `queue.wait()` 等同步点收集此前提交的 kernel；
- 同一物理时间步的全部 replica kernel 构成一个调度 batch；
- daemon 从 buffer accessor 的 RAW/WAR/WAW 关系构造 DAG；
- HEFT 为每个 kernel 选择单设备 placement 或安全 Split；
- profile 按 kernel requirement/shape、设备和 part 数更新。

### 当前 Split

- 主要沿 NDRange dimension 0 切 work-item；
- 写区间必须能够表示为连续的 dim0 owned blocks；
- read/read_write 在其他 part device 上使用完整 accessor snapshot；
- 每个 part 只 merge 自己的 owned write range；
- canonical merge device 才是 Split producer 的完整 current version；
- 宽 DAG guard 可以在 task parallelism 足够时拒绝 Split；
- `read_write` 本身不是禁区，只要写集合是 partition-disjoint 且 merge 只覆盖 owned range。

### 运行时能够看到什么

运行时能够看到：

- buffer identity；
- accessor mode、整体 range、元素大小和 memory range；
- kernel NDRange；
- DAG producer/consumer；
- device/part profile；
- 当前完整版本位置的粗粒度状态。

运行时看不到：

- 任意 work-item 对 read accessor 实际访问了哪些元素；
- `item[i]` 和邻居索引之间的仿射或间接关系；
- stencil radius；
- SPH neighbor list 中存放的动态索引值；
- 一个 execution chunk 对 read buffer 的精确 footprint。

Benchmark 不能假装运行时知道这些信息。

## 为什么纯运行时不能把通用 stencil Split 变成 halo Split

规则 stencil 的理想通信是：每个 part 保留自己的 interior，只交换半径 `R` 的边界 halo。对体积为 `V` 的三维网格，理想通信接近表面积量级，而不是复制整个 `V`。

但标准 SYCL kernel 在当前 handler 中只提供整体 accessor range。即使 kernel 代码实际执行：

```text
out[z,y,x] = f(in[z-R:z+R, y-R:y+R, x-R:x+R])
```

运行时也只知道 `in` 被读、`out` 被写，以及 NDRange 的大小。仅凭这些信息，把某个 part 的 read 缩成 `owned + halo` 是不安全的，因为同样的元数据也可能对应：

- periodic wraparound；
- transpose；
- arbitrary gather；
- indirect index array；
- 每个 work-item 读取完整 buffer；
- 数据相关的远距离读取。

因此纯运行时可以安全实现的改进是：

1. 为只读或完整 current replica 建立 per-device version/epoch，避免相同版本重复复制；
2. 对连续 Split 链保持相同 partition/device 对应关系；
3. 用 full-replica all-gather 取代每次回到单一 canonical device 后再重建 secondary snapshot；
4. 合并多个连续 field copy，优先 direct P2P；
5. 用 event dependency 连接 copy→kernel→merge，减少 host wait；
6. 只有真实 consumer 需要单一完整版本时才 materialize canonical merge；
7. 对应用显式声明的较小 accessor region 只移动该 region。

这些改进可能把常数降低约 1.5–2 倍，也能明显改善多 consumer 的 replica reuse，但其通信仍是 `O(V)`，不能自动变成 `O(halo)`。若要得到精确 halo，至少需要以下一种超出当前纯运行时边界的能力：

- 应用或库提供 execution-chunk→read-region metadata/range mapper；
- DPC++ 编译器对 accessor indexing 做可靠的仿射分析并生成 footprint；
- 编译器或后端插桩，在受控运行中学习访问集合；
- 新的显式 stencil/partition API。

第一种本质上引入了与 Celerity range mapper 相似的数据合同；后续仍可由本系统 profile 决定是否 Split，但不能再声称 halo 是由现有 runtime metadata 自动推导的。

所以当前论文/实验路线应当是：继续把完整 replica/version、异步 P2P 和延迟 materialization 作为通用 Split 改进，但不把规则 stencil 当作本系统在 Layer C 上优于 Celerity 的主正例。

## 科学模型：多副本 WCSPH

每个 replica `r` 包含 `N_r` 个等质量粒子，位于周期边界的三维盒子中。邻居候选使用带 skin 的 Verlet list。正式计时覆盖的步数必须保证最大位移不超过 `skin/2`，因此计时区间内不重建 neighbor list；运行结束时验证该条件。

推荐使用 Wendland C2 kernel。令 `q=r/h`：

```text
W(q,h) = alpha/h^3 * (1-q/2)^4 * (2q+1),  0 <= q < 2
W(q,h) = 0,                               q >= 2
```

实现中还需要它的径向梯度和用于黏性项的平滑 Laplacian。可以使用数值稳定的展开多项式，不能通过无物理意义的 `repeat` 循环制造计算量。

### 密度

```text
rho_i = m * W(0,h) + sum_{j in neighbors(i)} m * W(|x_i-x_j|, h)
```

### Tait 状态方程

```text
p_i = B * ((rho_i / rho0)^gamma - 1)
B   = rho0 * c0^2 / gamma
```

推荐 `gamma=7`，不同 fidelity class 可以使用不同 `c0`、`rho0` 和 `h`。

### 压力加速度

```text
a_i^p = -sum_j m * (p_i/rho_i^2 + p_j/rho_j^2) * grad W_ij
```

### 黏性加速度

```text
a_i^v = nu * sum_j m * (v_j-v_i)/rho_j * laplacian W_ij
```

### 积分

```text
v_i(t+dt) = damping * (v_i(t) + dt * (a_i^p + a_i^v + g))
x_i(t+dt) = periodic_wrap(x_i(t) + dt * v_i(t+dt))
```

这是简化但自洽的弱可压缩 SPH 时间推进。不同 replica 之间没有物理依赖；每个 replica 内的状态严格按时间步递推。

## 数据结构和内存几何

使用 trivially-copyable 的 16-byte 向量，避免依赖扩展数学类型：

```cpp
struct Vec4 {
  float x, y, z, w;
};
```

每个 replica 私有持有：

| Buffer | 类型/大小 | 用途 |
|---|---|---|
| `position[2]` | `Vec4[N]` ×2 | ping-pong 位置 |
| `velocity[2]` | `Vec4[N]` ×2 | ping-pong 速度 |
| `density` | `float[N]` | 当前密度 |
| `pressure` | `float[N]` | 当前压力 |
| `pressure_accel` | `Vec4[N]` | 压力分支输出 |
| `viscosity_accel` | `Vec4[N]` | 黏性分支输出 |
| `neighbor_count` | `uint32_t[N]` | 每粒子真实候选数 |
| `neighbor_index` | `uint32_t[N*Kmax]` | 固定 stride Verlet rows |

全部 buffer 都是一维 row-major。粒子 `i` 的 neighbor row 是：

```text
[i*Kmax, i*Kmax + neighbor_count[i])
```

每 replica 的近似 bytes 为：

```text
N * (108 + 4*Kmax)
```

例如 `N=131072, Kmax=192` 时约为 114.8 MB/replica；64 个同规模 replica 约 7.35 GB。正式 mixed 配置的平均 `N` 会更小，但必须用 64-bit checked arithmetic 在程序启动时输出真实估算，并在溢出或超过用户上限时失败，不能静默缩小。

所有 replica 必须有独立 buffer identity。禁止为了节省代码把所有 replica 拼成一个大 buffer 后提交一个单 kernel，因为那会把真实的 runtime task parallelism隐藏进 kernel NDRange。

## Kernel 顺序、数据读取与依赖

每个物理时间步对所有 replica 按层提交，不能逐 replica 提交完整链后再提交下一个 replica。正确顺序是先提交全部 `Density`，再全部 `EOS`，再两个 force 分支，最后全部 `Integrate`。

### `Density[r,t]`

NDRange：

```text
range<1>(N_r)
```

读取：

- `position[cur]`；
- `neighbor_count`；
- `neighbor_index`。

写入：

- `density[i]`，`discard_write`。

每个 work-item 遍历自己的 neighbor row，执行 periodic minimum-image、距离、Wendland kernel 和累加。只写 `density[i]`。

### `EOS[r,t]`

读取：

- `density[i]`。

写入：

- `pressure[i]`，`discard_write`。

每个 work-item 执行 Tait EOS、正密度保护和有限值保护。这是较轻 kernel，但它是压力分支的真实物理阶段，不能人为删除依赖或加入重复循环。

### `PressureForce[r,t]`

读取：

- 完整 `position[cur]`；
- 完整 `density`；
- 完整 `pressure`；
- owned `neighbor_count/neighbor_index` rows。

写入：

- `pressure_accel[i]`，`discard_write`。

每个 work-item 遍历 neighbors，计算对称压力项和 `grad W`。只写自己的 acceleration。

### `ViscosityForce[r,t]`

读取：

- 完整 `position[cur]`；
- 完整 `velocity[cur]`；
- 完整 `density`；
- owned `neighbor_count/neighbor_index` rows。

写入：

- `viscosity_accel[i]`，`discard_write`。

该 kernel 不读取 `pressure`，因此在 `Density` 完成后可以与 `EOS -> PressureForce` 分支并行。

### `Integrate[r,t]`

读取：

- `position[cur][i]`；
- `velocity[cur][i]`；
- `pressure_accel[i]`；
- `viscosity_accel[i]`。

写入：

- `position[next][i]`；
- `velocity[next][i]`，均为 `discard_write`。

它执行阻尼积分、周期 wrap，并产生下一物理时间步的 current state。

### 单 replica DAG

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

依赖为：

- `Density -> EOS`：RAW on density；
- `Density -> PressureForce`：RAW on density；
- `Density -> ViscosityForce`：RAW on density；
- `EOS -> PressureForce`：RAW on pressure；
- `PressureForce/ViscosityForce -> Integrate`：RAW on two accelerations；
- `Integrate[t] -> Density/force[t+1]`：RAW on new position/velocity；
- ping-pong reuse产生必要 WAR/WAW ordering。

不同 replica 不共享任何 buffer，因此没有跨 replica 依赖。

## 逻辑并行度要求

设 replica 数为 `R`：

```text
kernels_per_step       = 5R
maximum_structural_width = 2R
critical_path_levels     = 4
equal-cost structural average parallelism = node_count / critical_path = 1.25R
```

默认 `R=64`：

```text
kernels_per_step = 320
max_width        = 128
equal-cost structural average parallelism = 80
width / 2 GPUs   = 64 at the widest level
```

推荐正式规模在显存允许时使用 `R=96`：

```text
kernels_per_step = 480
max_width        = 192
```

这才属于“逻辑并行度极大”。不能只依赖第一层短暂 fan-out；每个层级至少有 `R` 个 ready task，压力/黏性分支达到 `2R`，并且该形态每个物理时间步重复。

逻辑宽度还必须按实测 work 加权。正式规模要求三种邻居重 kernel的 native 单卡第 25 百分位至少为 3 ms，中位数建议为 5–20 ms。若不满足，应增加真实 `N`、neighbor density 或 `Kmax`，不能加入无意义 repeat。

## Fidelity classes 和 profile 可学习性

默认使用 8 个 class，每个 class 至少有 8 个 replica。建议：

```text
N_c = align_down(base_particles * (100 - c*coarsen_percent) / 100, 256)
```

同时让 class 控制：

- smoothing length；
- reference density；
- viscosity；
- sound speed；
- 初始化粒子密度；
- 平均 neighbor count。

同一 class 的 shape 和物理参数必须相同，从而可以共享 profile。不同工作量 class 必须至少在 `N` 或 neighbor buffer shape 上不同，使当前 profile key 能够区分。禁止创建“完全相同 requirement/shape、只靠隐藏 scalar 改变数倍循环次数”的正面案例，因为当前 profile key不能区分这种 workload。

推荐默认值：

```text
replicas=64
particles=131072
steps=30
max_neighbors=192
classes=8
coarsen_percent=4
mode=mixed
```

这些只是初始候选。正式参数必须通过 native profiling 和显存估算确定，而不是把默认值直接当论文规模。

## 邻居表初始化与物理有效性

初始化不计入 `run_sec`。推荐在 host 上：

1. 构造周期三维晶格/低扰动粒子分布；
2. 用 cell list 找到 `cutoff + skin` 内的 Verlet candidates；
3. 填充固定 stride neighbor rows；
4. 若任何粒子超过 `Kmax`，输出错误并退出；
5. 保存初始位置用于最终 displacement 验证。

可以每个 fidelity class 只构造一次初始几何和 neighbor topology，再复制到该 class 的 replica-private buffers；replica 通过物理参数和初始速度扰动形成不同轨迹。这样减少 host 初始化计算，但设备上的 buffer identity 仍然完全私有，不能让多个 replica 共享一个会在设备间迁移的 SYCL neighbor buffer。

不能用随机任意索引伪造 neighbor list。允许不同 class 使用不同初始密度和规则扰动，但每个 neighbor index 必须对应真实空间候选。

为避免在 timed region 中加入复杂 neighbor rebuild，选择足够小的 `dt`、速度和步数，使最终：

```text
max_particle_displacement < skin / 2
```

若不满足，`VERIFY passed=0`，不能把物理失效的运行作为性能结果。

## 同步窗口

默认一个完整 SPH 物理时间步是一个 wait window：

1. 提交全部 replica 的五类 kernel；
2. 不在 kernel 之间 wait；
3. 时间步末执行一次 `queue.wait()`；
4. 下一步使用前一步 profile。

这是重复物理积分边界，也为 runtime 提供稳定的 profile-corrected window。`--wait-each-kernel 1` 仅用于排错，不能用于正式性能。

可以提供 `--window-steps`，但默认必须为 1。大于 1 时只是在一个 batch 中连接多步 DAG，用于图规模压力测试；正式主结果应优先采用每步更新 profile 的默认模式。

## Split 安全性和预期决策

所有 kernel 使用 `range<1>(N_r)`，work-item `i` 只写一维输出的元素 `i`。因此沿 dim0 切分时：

- `N_r % num_parts == 0`；
- 每个 part 写连续粒子区间；
- 不同 part 的 writes 不重叠；
- neighbor reads 可以跨任意 part；
- 当前完整 read snapshot 语义保证正确；
- merge 只覆盖每个 part 的 owned output range。

但默认 `R=64/96` 时，task parallelism 远大于 GPU 数。正确的主模式决策应是：

```text
selected split kernels = 0 or extremely close to 0
```

两张 GPU 应分别执行不同 replica 的 whole kernels，并让同一 replica 的后继靠近 producer。Split 能力在这里用于证明 scheduler 会拒绝不盈利的合法 Split，而不是要求每个大 kernel 都 Split。

提供 `--mode single-large`：只保留一个超大 replica，用于控制。该模式 ready width 很低，Celerity 的 deterministic distribution 可能优于当前系统的 full-snapshot Split；这个结果必须如实报告。

## 当前系统上的预期数据移动

每个 replica 的所有 buffer 都是私有的。首次分配后，理想 steady state 为：

```text
replica A state -> GPU 1 resident
replica B state -> GPU 2 resident
same replica successors -> same GPU
cross-step full-state D2D -> approximately zero
host materialization -> final samples/check only
```

允许冷启动时从 host 到首次设备的 H2D，也允许 profile 尚未稳定时少量迁移。step 5 以后 ordinary D2D/H2D/D2H 应接近零或只在 placement 真正改变时出现。

如果当前 runtime 每一步仍把同一 replica 的私有 state 在两卡之间移动，应先修 placement/residency；不能修改 bench 加 fake dependency 或手工设备副本来隐藏问题。

## Celerity 的正确映射

必须使用准确 mapper，不能用故意错误的 `all` 惩罚 Celerity。

### Density

- execution：particle interval；
- `neighbor_count`：one-to-one；
- `neighbor_index`：自定义 slice，将 chunk `[b,e)` 映射为 `[b*Kmax,e*Kmax)`；
- `position`：`all`。

`position=all` 是语义所需，而非 mapper 偷懒。neighbor indices 存放在 buffer 内容中，普通 range mapper 在 command graph 生成时不能根据这些动态值返回一组任意散点；安全上界是整个 position buffer。

### EOS

- density：one-to-one；
- pressure：one-to-one。

### PressureForce

- owned neighbor rows：slice；
- position、density、pressure：`all`；
- pressure acceleration：one-to-one。

### ViscosityForce

- owned neighbor rows：slice；
- position、velocity、density：`all`；
- viscosity acceleration：one-to-one。

### Integrate

- position/velocity/两个 acceleration：one-to-one；
- next position/velocity：one-to-one。

若 Celerity 把一个 replica task 分到两张 GPU，`Integrate` 会产生分区的新 position/velocity；下一步 `Density`/force 的 `all` read 又要求每个参与 worker 看见完整 current state。因此每个 replica 每步都可能出现全局可见性维护。这是间接粒子算法本身与 deterministic per-task distribution 的交互，不是错误 mapper。

必须检查所用 Celerity 版本是否支持：

- task hints；
- split constraints；
- worker/device subset；
- 独立 task 并发；
- command/instruction graph 优化；
- collective pattern recovery。

如果可以把每个 replica 固定到单 worker，应启用并报告。此时比较变为：固定/用户指定 replica placement 对 profile-guided、device-aware global placement，而不能继续声称 Celerity 必然分布每个 task。

最终 Celerity 对手取默认准确 mapper 与版本支持的最佳 hints 两者中更快的一项。主 `mixed` 模式只有在正确性一致、相同双 GPU、相同物理工作量下仍达到下文规定的速度门槛，才算本系统胜出；不能只与较慢的 Celerity 默认配置比较。

## 三层研究假设

### Layer A：计算成本

相同 execution-range 长度不完全决定 SPH 成本。实际成本还受：

- `neighbor_count[i]` 分布；
- 粒子密度与 cutoff；
- 分支发散；
- pressure/viscosity 参数；
- 设备当前有效速度。

本系统应通过 class/device profile 学到重 kernel 成本，并让预测 window time 收敛。

确认指标：raw profile、EWMA、每 class/device 误差、预测与实际 window time。拒绝条件：profile key 混合不同隐藏 workload，或预测长期不收敛。

### Layer B：调度决策

当有 64–96 个独立 replica 时，两卡各执行不同 whole-replica tasks，通常比每个 task 同时占用两卡更有吞吐优势。HEFT 应联合考虑：

- class cost；
- device speed；
- device available time；
- producer affinity；
- 是否使用第二张卡；
- 是否拒绝 Split。

确认指标：每卡 assigned predicted/actual work、active overlap、window makespan、placement stability、Split 数和最佳 co-located 对照。拒绝条件：任务数很多但第二张卡仍长期空闲，或每步 placement 大幅抖动。

### Layer C：数据维护

本系统的主假设不是比 Celerity 更擅长 halo，而是：对大量私有、间接 gather 的 replica，whole-pipeline placement 可以避免让每个 replica 的 current state 在所有设备上可见。

确认指标：每 step ordinary D2D/H2D/D2H bytes、Celerity copy/push/await/collective bytes、backing allocation 和 command count。拒绝条件：本系统仍每 step 移动完整私有 state，或 Celerity 在公平 hints 下同样实现零移动且调度成本相当。

## 运行模式

至少提供：

```text
--mode mixed
--mode uniform
--mode single-large
```

- `mixed`：主正面模式，8 个 fidelity class，验证 profile-guided weighted placement；
- `uniform`：所有 replica 同规模同参数，控制 size/profile 异构性；
- `single-large`：一个超大 replica，Celerity-friendly distributed control。

其他参数：

```text
--particles
--replicas
--steps
--max-neighbors
--classes
--coarsen-percent
--dt
--smoothing-length
--skin
--window-steps
--wait-each-kernel 0|1
--host-read-full 0|1
--verify 0|1
```

所有参数必须做范围、乘法溢出、`N % split_parts` 相关和内存估算检查。

## 实验矩阵

同一参数至少运行：

| 编号 | 配置 | 目的 |
|---|---|---|
| A0 | native SYCL，物理 GPU 0 | 最佳单卡候选 |
| A1 | native SYCL，物理 GPU 1 | 设备校准 |
| B1 | offline 强制 device 1，no Split | 控制面成本 |
| B2 | offline 强制 device 2，no Split | 设备/queue 路径 |
| C | offline 双 GPU，强制 no Split | whole-replica placement 主对照 |
| D | offline 双 GPU，默认 selective Split | 检查默认是否正确选择零/极少 Split |
| E | Celerity，准确 mapper，默认策略 | 默认行为及 visibility/distribution 成本 |
| F | Celerity，启用版本支持的最佳 hints | 最强公平对手；与 E 中更快者共同决定 Celerity 基线 |
| G | `single-large` 三系统对比 | Celerity-friendly 控制 |

每组至少 5 次，报告 median 和 min；详细 trace 关闭。单次日志不是最终性能证据。

## 通过门槛

主 `mixed` 模式只有同时满足以下条件，才能作为正面结果：

1. 小规模 CPU reference 或高精度 host reference 通过；
2. native、offline 和 Celerity 的结果在定义容差内一致；
3. `R>=64`，最大宽度至少 128；
4. Density/PressureForce/ViscosityForce 的 native 第 25 百分位至少 3 ms；
5. offline 双 GPU no-Split 相对最佳 native 单卡至少 1.3×；
6. offline 双 GPU相对 offline 最佳单卡至少 1.3×，证明收益不是 native/offline 差异；
7. 两卡在大部分 `run_sec` 中有重叠工作；
8. step 5 后私有 state 不发生每步全量跨卡移动；
9. 默认 selective 策略不劣于强制 no-Split，且能解释每个保留 Split；
10. 相对 E/F 中更快的公平 Celerity 配置，满足 `speedup_over_celerity = median(Celerity run_sec) / median(offline run_sec) >= 1.20`；
11. 对 Celerity 的优势能由实测 placement、visibility/data movement 和 cost decision 归因，而不是错误 mapper、减少其 worker 或禁用对方功能；
12. `single-large` 结果单独报告，即使 Celerity 更快也不能隐藏；
13. 最终位移小于 `skin/2`，所有值 finite，`VERIFY passed=1`。

如果第 4 条失败，扩大真实粒子数/邻居数。如果第 5–8 条失败，应先修 runtime，不能通过减少 Celerity worker、增加 fake work 或修改访问语义来制造收益。如果第 10 条失败，则这个候选至多是系统调度测试，不能作为“本系统优于 Celerity”的正面 miniapp；应分析 Celerity 已消除的开销或本系统缺失的调度能力后再决定换 bench 还是改 runtime。

## 验证设计

### 小规模 reference

对少量 replica、`N<=2048`、少量 steps，在 host 上以相同公式顺序执行：

- density；
- EOS；
- pressure force；
- viscosity force；
- integrate。

比较固定粒子样本以及全量 L1/L2 checksum，允许合理 float tolerance。

### 正式规模验证

正式规模在 `run_sec` 结束后读取最终 position，用于严格检查最大位移和周期边界；这部分必须计入 `host_sec` 而不能混入 `run_sec`。其余大场默认只读取每 replica 的固定粒子样本，并检查：

- position/velocity/density/pressure finite；
- density positive；
- position 在周期盒范围内；
- checksum finite；
- max displacement `< skin/2`。

`--host-read-full 1` 额外遍历 velocity、density、pressure 和 acceleration 全场，用于调试/reference。无论该开关取值如何，所有 host materialization 都在 `run_sec` 之后并单独计入 `host_sec`。

## 输出格式

输出至少包含：

```text
CONFIG particles=... replicas=... steps=... max_neighbors=... classes=... mode=...
DAG kernels_per_step=... max_width=... critical_path_levels=... total_kernels=...
CLASS id=... replicas=... particles=... avg_neighbors=... h=... viscosity=... dt=...
MEMORY bytes_per_class=... total_estimated_bytes=...
TIMING init_sec=...
TIMING queue_sec=...
TIMING run_sec=...
TIMING host_sec=...
TIMING total_sec=...
RESULT checksum=... sample_rho=... sample_pos=... max_displacement=...
VERIFY passed=1
```

初始化、neighbor-list construction、queue creation、timed run 和 host read 必须分开。

## 实现约束

- C++17；
- `#include <sycl/sycl.hpp>`；
- 只使用标准 SYCL buffer/accessor/queue；
- 不依赖 USM、MPI、oneMKL、外部 solver 或外部数据文件；
- 使用命名 kernel class；
- normal mode 只在物理时间步末 wait；
- 不使用 reduction、atomic 或 host scalar 决策进入 timed DAG；
- 不使用 fake guard buffer 编码调度；
- 不把 replica 合并成一个大 NDRange；
- 不手工选择 GPU 或复制 device buffers；设备选择交给 runtime；
- 不用无意义 repeat、busy loop 或 transcendental spam 增加计算量；
- 用真实 neighbor interactions 调整粒度；
- 写 accessor 必须与 item-wise owned write 合同一致；
- 所有 byte/element arithmetic 使用 `size_t`/`uint64_t` 并检查溢出；
- 正式计时关闭 runtime detailed trace。

## 明确禁止的错误简化

不能把本应用改成：

- 一个 kernel 同时处理全部 replica；
- 每 replica 只有一个互不依赖的 element-wise kernel；
- 随机 neighbor indices；
- 每个 kernel 后 wait；
- host 在每一步读回全量状态；
- 为了防止 Split 添加假依赖；
- 为了让本系统赢而给 Celerity 使用错误 `all`/错误 one-to-one mapper；
- 为了放大成本而隐藏修改物理工作的 scalar，使 profile key 无法区分；
- 只报告最有利模式而隐藏 `uniform` 或 `single-large`。

## 实现完成后的必备说明

生成源码后，必须同时提供一份简洁说明，逐项回答：

1. 五类 kernel 的顺序和物理含义；
2. 每个 kernel 读取/写入哪些 buffer；
3. 完整 DAG、最大宽度和关键路径；
4. 为什么默认应 whole-replica placement 而不是 Split；
5. 当前 Split 为什么正确但对 indirect gather 昂贵；
6. Celerity 每个 accessor 的准确 mapper；
7. 三层假设分别由哪些 counter/timeline 验证；
8. 哪些结果是实测，哪些仍是待验证假设；
9. 编译和运行命令；
10. native、offline、Celerity 的公平实验表。

本文档的成功标准不是“代码能够运行”而是：新 miniapp 在不触及当前系统边界的前提下，真正提供数十到上百个可调度的重计算 pipeline，让 wait-window DAG、profile-guided whole-kernel placement、producer affinity 和粗粒度数据驻留产生可以测量、可以归因的收益，并在主 `mixed` 模式中明确胜过最佳公平 Celerity 配置；`single-large` 则保留为允许 Celerity 占优的诚实边界控制。
