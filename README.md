# conv5

5 层 3×3 int8 CNN 推理,RTX 4090 Laptop (SM89),仅用 `mma.sync` + `ldmatrix`(无 cp.async)。

## 模型

conv(4→16)+ReLU → conv(16→32)+ReLU → conv(32→16)+ReLU → conv(16→16)+ReLU → conv(16→4) → NHWC 输出
输入 1×4×360×796 fp32 NCHW,输出 1×360×796×4 fp32 NHWC(transpose 融合在 conv5 epilogue)。

## 构建与运行

```
powershell build.ps1        # CUDA 12.1 + VS2019 工具链, sm_89
build\conv5.exe test        # fragment 微测试 + 三实现逐层 bit-exact 验证
build\conv5.exe bench       # 计时(3 warmup + 20 次平均)
build\conv5.exe gen         # 重新生成 weights.bin / input.bin
```

## 实测结果(RTX 4090 Laptop,CC 8.9)

| 实现 | L0 | L1 | L2 | L3 | L4 | total | 有效带宽 |
|---|---|---|---|---|---|---|---|
| v1 naive | 0.904 | 1.801 | 1.789 | 0.905 | 0.232 | 5.632 ms | 9.8 GB/s |
| v2 smem 分块 | 0.573 | 1.134 | 1.201 | 0.581 | 0.180 | 3.669 ms | 15.0 GB/s |
| **v3 mma+ldmatrix** | **0.043** | **0.089** | **0.105** | **0.043** | **0.033** | **0.313 ms** | **175.6 GB/s** |

- v3 相对 v1 提速 **18×**,相对 v2 提速 **11.7×**
- 三个版本与 CPU int8 参考**逐层 bit-exact**(整型累加无舍入,验收为精确相等)
- vs CPU fp32 参考:平均绝对误差 0.023(参考输出均值 0.904,平均相对误差 ≈ 2.6%,per-tensor 对称量化的固有噪声水平)

## v3 kernel 要点(src/conv_mma.cu)

- implicit GEMM:M=286560(NPQ) × N=C_out × K=C_in×9;K 排序 `gemm_k = c + C_in*(s+3r)`,每个 k16 块 = 同一 (r,s) 的 16 个连续通道 = NHWC 中 16B 连续向量
- block tile 128 行 × 全部输出通道;4 warps 沿 M 划分;`mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32`
- 权重整块常驻 smem(n 主序,K 连续;K=288 时行距 pad 到 304B=19×16B 保 ldmatrix 无 bank conflict)
- A 双缓冲(LDG→寄存器→STS,SM75 风格,无 cp.async):先发下一 k-step 的 LDG,再对当前 buffer ldmatrix+mma,再 STS 到另一 buffer
- `ldmatrix.x4` 装 A(32 行),`ldmatrix.x1/x2/x4` 装 B(N/8 个 n8 tile);s8 打包:b16 = 2×int8
- epilogue:bias + requant(`__float2int_rn(acc*mult)` clamp[0,127],ReLU 折叠进下界)+ 散写;L4 直接 dequant 写 fp32 NHWC
- 资源占用:32-56 寄存器 / 5-9KB smem / 零 spill

## 踩坑记录

1. `ldmatrix` 是 warp 集体指令,放入 `if (lane < 8)` 分支会死锁——所有 32 lane 必须执行,仅低 lane 提供有效地址
2. `ldmatrix.xN` 地址取自 lane 0..8N-1(x2 用 lane 0-15),高 lane 地址被忽略,映射错会静默加载重复数据
3. int8 m16n8k16 的 B fragment(n 主序存储)用**非转置** ldmatrix 即可,等价于 `*(u32*)&Bs[lane/4][(lane%4)*4]`(由 src/mma_test.cu 微测试实证)
4. CIN=32 时每个 (r,s) 占 2 个 k16 步,LDG 需加通道块偏移 `(ks % (CIN/16)) * 16`

## 目录

```
build.ps1        一键构建
tools/gen_weights.cpp   权重生成 + 校准式量化(模拟 int8 链路逐层定 scale)+ .bin IO
src/quant.h      常量与数据结构
src/reference.*  CPU fp32 / int8 参考
src/mma_test.cu  mma/ldmatrix fragment 布局微测试
src/conv_naive.cu   v1
src/conv_tiled.cu   v2
src/conv_mma.cu     v3
src/main.cpp     验证 + 计时框架
```
