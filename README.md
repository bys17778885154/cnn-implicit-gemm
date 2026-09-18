# cnn-int8

5 层 3×3 int8 CNN(输入 1×4×360×796)在 RTX 4090 Laptop (SM89) 上的两种 GPU 实现,
单仓库、双实现、零耦合:两种实现互不引用,只共享 `common/` 的模型层
(量化/CPU 参考/权重生成),保证 bit-exact 验收用同一把尺子。

## 结构

```
common/            共享层:quant.h(常量/模型)、reference.cpp(CPU fp32+int8 参考)、
                   gen_weights.cpp(校准式量化 + .bin IO)——两个实现唯一的公共依赖
cuda/              CUDA 实现(mma.sync + ldmatrix,无 cp.async),独立构建运行 → conv5.exe
vulkan/            Vulkan 实现(VK_KHR_cooperative_matrix),独立构建运行 → vkconv5.exe
bench/             CUTLASS 对照基准 + 大通道扫描(sweep)
```

## 独立构建与验证(各自目录内完成,互不依赖)

```
cd cuda;    powershell build.ps1; .\build\conv5.exe   test / bench / benchcmp / dump
cd vulkan;  powershell build.ps1; .\build\vkconv5.exe test / bench / benchmany / dump
cd bench;   nvcc ... sweep.cu                       # CUTLASS 对照(见 bench/README.md)
```

工具链互不相交:cuda = nvcc(sm_89),vulkan = glslc + cl(Vulkan SDK)。
weights.bin/input.bin 各目录各持一份(确定性生成,内容逐位一致)。

## 性能汇总(RTX 4090 Laptop,SM89)

### CUDA 版本演进(逐层 event 口径,`cuda`)

| 版本 | total | 有效带宽 | 相对 v1 |
|---|---|---|---|
| v1 naive(每线程一像素) | 5.63 ms | 9.8 GB/s | 1× |
| v2 smem 分块直接卷积 | 3.65 ms | 15.1 GB/s | 1.5× |
| v3 mma k16(m16n8k16 + ldmatrix + 双缓冲) | 0.309 ms | 178 GB/s | 18× |
| **v3 mma k32(m16n8k32)** | **0.279 ms** | **198 GB/s** | **20×** |

### Vulkan 版本演进(coopmat,`vulkan`)

| 阶段 | 耗时 | 说明 |
|---|---|---|
| 初版(全标量 staging,逐层 submit) | 1.352 ms | 41 GB/s |
| kernel 优化(int32 向量加载 + i8vec4 存储 + 寄存器双缓冲 + BM=192) | 0.496 ms | staging 向量化贡献 ~2.3× |
| + 单 cmdbuffer 合并(5 dispatch + 层间 barrier) | 0.47 ms | 端到端 wall 0.77→0.47ms |
| + epilogue 整行 scratch(峰值 shared 41→21KB,40KB 约束内) | **0.40-0.43 ms** | 写回合并度不变,占用率兑现 |

### 跨实现最终对比(四方口径,交替背靠背同窗口,20 次平均 × 3 轮取中位)

| 路径 | GPU 吞吐 | 端到端 wall |
|---|---|---|
| CPU(参考实现,OpenMP) | — | ~秒级 |
| CUDA mma32 逐层 event 同步 | 0.24-0.28 ms | 0.31 ms |
| **CUDA mma32 异步连发**(100 链/1 次同步) | — | **0.29-0.30 ms** |
| Vulkan coopmat 逐层 submit | 0.42 ms | 0.65 ms |
| **Vulkan coopmat 合并 cmdbuffer**(100 链/1 次提交) | 0.40-0.43 ms | **0.40-0.43 ms** |

结论:
- **CUDA 全口径领先约 1.5×**:CUDA kernel 异步发射 ~2-5µs/个;Vulkan 每 dispatch
  ~80µs 固定成本(驱动 pipeline 切换/barrier,与 SM 频率无关)——API 层调度开销,
  非 shader 差距
- **三方逐位一致**:CPU / CUDA / Vulkan 最终输出 SHA256 相同(`0DA656256659892F...`,
  1,146,240 个 fp32),dump 模式可随时复验
- shared memory:CUDA 峰值 ~12.8KB(累加器在寄存器);Vulkan 峰值 21KB——均在 40KB 约束内
- 测量条件注记:笔记本 GPU 时钟有波动,跨实现数字一律交替背靠背同窗口采集

## CUTLASS 对照与选型决策(bench/)

用手写 kernel 与生产级 CUTLASS(6 配置扫描 × 6 形状,2032 点强验证)对比后,
得到本仓库最重要的工程结论——**按 C_out 选实现**:

| C_out | 胜者 | 倍数 | 机制 |
|---|---|---|---|
| 16 | **手写** | 4.4× | tile 精确贴合通道数,B 常驻 smem,fused requant |
| 32 | **手写** | 1.7-2.2× | 同上,优势开始收窄 |
| **~48** | **交叉点** | — | 两侧斜率都陡 |
| 64 | CUTLASS | 3.2× | 2D tile 的 A/B 双复用开始兑现(util 48%) |
| 128 | CUTLASS | 10.6× | 深入 compute-bound(util 97%) |
| 256 | CUTLASS | 14.6× | util 114%(boost 超标称) |

本质是两个架构各自的甜蜜点:
- **手写**:B 整块常驻 smem、N 维不切分——C 小时完美贴合;C 大时 B 装不进 smem,
  强行 N-分块会把激活重复搬 C/CHUNK 遍,必输
- **CUTLASS**:128×128 的 2D tile,A/B 同时复用,复用率与 C 无关地稳定;
  但对 C<64 全是 padding 浪费,且无 int8 requant 融合

**一句话规则**:C_out ≤ 32 的"瘦"卷积(量化后第一层、点云/深度图头、部分检测头)
自己写;C_out ≥ 64 的标准卷积(ResNet/ConvNet 主体)直接用 CUTLASS/cuDNN。
本仓库 5 层 C_out=4~32,整体落在手写侧,所以手写全面获胜——是负载属性,不是手写普遍更强。

另外:CUTLASS 全层 raw-s32 bit-exact(与 CPU 精确整数累加),第三次交叉验证了
本项目参考实现的正确性。详见 `bench/README.md` 与 `docs/tensor-utilization.md`。

## 历史

本仓库由 conv5 与 vkconv5 两个仓库经 `git subtree` 合并而成,双方全部提交历史保留
(`git log --follow common/quant.h` 可追溯至原仓库)。
