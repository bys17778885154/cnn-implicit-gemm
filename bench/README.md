# CUTLASS int8 conv2d fprop 对照基准(cnn-int8/bench)

用 CUTLASS(生产级 implicit GEMM 库)的 SM80 s8 conv kernel 跑我们的 5 层网络,
与手写 conv5(mma.sync + ldmatrix)同数据、同协议对比。

## 配置

- 来源:CUTLASS 单测 `conv2d_fprop_implicit_gemm_s8nhwc_..._sm80.cu` 的已知良好配置
- kernel:`DefaultConv2dFprop<s8, NHWC, s8, NHWC, s32, OpClassTensorOp, Sm80, m16n8k32>`
  主循环 `ImplicitGemmMultistage`(cp.async 3-stage;即被本项目禁用的路线,作参照)
- 两种 tile:官方默认 `128×128×64`(warp 64×64×64)与更贴合小通道的 `128×64×64`(warp 64×32×64)
- 数据:与 conv5 同一份 weights.bin/input.bin;A=NHWC int8(含 padding 通道),B 重排为
  CUTLASS 的 KRSC 内存序,C=raw s32(alpha=1,beta=0,无 requant)
- 验证:全部 5 层 raw s32 与 CPU 精确整数累加 **bit-exact**(比 conv5 的验证更原始一层)
- 计时:每层独立,3 warmup + 20 iter cudaEvent 平均(与 conv5 bench 同协议)

## 结果(RTX 4090 Laptop,同窗口背靠背)

| 层 | CUTLASS 128×128×64 | CUTLASS 128×64×64 | **conv5 mma32(手写)** |
|---|---|---|---|
| L0 (16→16) | 0.330 ms | 0.232 ms | **0.045 ms** |
| L1 (16→32) | 0.276 ms | 0.194 ms | **0.085 ms** |
| L2 (32→16) | 0.276 ms | 0.194 ms | **0.075 ms** |
| L3 (16→16) | 0.280 ms | 0.196 ms | **0.044 ms** |
| L4 (16→4)  | 0.278 ms | 0.195 ms | **0.031 ms** |
| **合计** | 1.44 ms | 1.01 ms | **0.28 ms** |

## 分析:为什么手写 kernel 快 3.6-5 倍

1. **tile 过配**:CUTLASS tensorop kernel 的 N(tile)最小实用粒度是 64/128 个输出通道,
   本网络 C_out=4~32 → N 维 padding 浪费 2-32×(128×128 配置最严重;64 列配置已好 30%,
   再小会掉出 tensorop 高效路径)。conv5 的 N tile = 精确 C_out
2. **输出位宽**:CUTLASS 路径输出 raw s32(4B/元素),conv5 融合 requant 输出 int8(1B)
   ——写流量天然 4×;这是对比口径的固有差异(库不负责量化)
3. **通道感知特化**:conv5 按 5 层各自 (CIN,COUT) 模板实例化(寄存器精确分块、B 整块常驻
   smem);CUTLASS 是通用分块,无法为 16 通道这种极端形状做激进特化
4. 有趣的是 CUTLASS 用了 cp.async(3-stage)+ Multistage——被本项目禁用的路线——
   仍然慢于 LDG→reg→STS 的手写版:**访存模式与 tile 匹配度比流水线技术更重要**

## 结论

- 本网络(C_out ≤ 32 的"瘦"卷积)是通用 GEMM 库的死角,定制 kernel 收益 3.6-5×
- 反过来,若换 C_out ≥ 128 的标准卷积,应直接用 CUTLASS/profiler 搜配置——
  那是它的主场,手写没有优势
- CUTLASS 全层 bit-exact 通过也再次交叉验证了本项目 CPU 参考与 GPU 实现的正确性

## 大通道卷积对照(sweep.cu,6 配置扫描 + 强验证 + 交错计时)

`sweep.cu` 对每个形状跑 **CUTLASS 6 配置**(3 tile × {Analytic, Optimized} 迭代器)+
**手写 N-分块变体**(conv_mma32 kernel 共享头 `conv_kernel.cuh` + koff 列偏移,chunk 16/32),
每配置 2000 随机点 + 32 边界/角落点精确验证,3 轮取最优。随机数据,raw s32 统一口径:

| 形状 | CUTLASS 最佳 | util | 手写 N-分块 | util | 胜者 |
|---|---|---|---|---|---|
| 16→16 | 0.167 (128×64 opt) | 3% | **0.038** | 15% | **手写 4.4×** |
| 16→32 | 0.152 (128×64 opt) | 8% | **0.091** | 13% | **手写 1.7×** |
| 32→16 | 0.100 (128×64 opt) | 12% | **0.045** | 26% | **手写 2.2×** |
| 64→64 | **0.195** (128×64 opt) | 48% | 0.632 | 15% | **CUTLASS 3.2×** |
| 128→128 | **0.384** (128×128 ana) | **97%** | 4.085 | 9% | **CUTLASS 10.6×** |
| 256→256 | **1.310** (128×128 opt) | **114%** | 19.2 | 8% | **CUTLASS 14.6×** |

**夯实后的结论**:
1. **交叉点在 C_out = 32~64 之间,比此前判断更锋利**:C_out≤32 手写赢 1.7-4.4×,
   C_out=64 CUTLASS 反超 3.2×(即使给手写加了 N-分块绕过 smem 上限)
2. **CUTLASS 最佳 tile 随形状移动**:瘦层 128×64+Optimized 迭代器,胖层 128×128
   (Ana/Opt 均可)——单配置对比会误判,扫描后结论稳定
3. **256→256 出现 util=114%**:2×286560×256×2304 OP / 1.31ms = 258 TOPS > 226 TOPS
   标称峰值(1455MHz×76SM×2048OP/clk/SM)——说明 GPU 短时 boost 超过标称口径,
   此前所有"util"数字含此口径不确定性(相对结论不受影响)
4. **手写 N-分块在大通道失效的根因**:按 chunk 串行发射 ×8/×16 次 kernel,A 激活被
   重复搬运 C_out/CHUNK 遍(64→64 即 2×,256→256 即 16×),完全没有 A 复用——
   这正是 CUTLASS threadblock 128×128 tile 内 A/B 同时复用的设计意义
5. **此前"架构无法实例化"修正为"架构不适合"**:N-分块可以绕过 smem/寄存器上限,
   但代价是 A 重复搬运,大通道下必然输给 2D tile 复用

## 复现
```
cd bench
powershell build.ps1        # 128x128x64 → build\cutlass_conv.exe
powershell build64.ps1      # 128x64x64  → build\cutlass_conv64.exe
build\cutlass_conv.exe
```
依赖:D:\b00852572\cutlass-src(git clone --depth 1 NVIDIA/cutlass)+ ../cuda/weights.bin。
大通道版:build\cutlass_large.exe / build\ours_large.exe(随机数据,无需模型)。

sweep: nvcc sweep.cu → build\sweep.exe(6 配置 + N-分块,需 cutlass-src)
conv_kernel.cuh:从 cuda/src/conv_mma.cu 生成的共享 kernel 头(RAW/koff 扩展,单一事实源)
ours_large / cutlass_large:单形状深挖版(64→64 等)
