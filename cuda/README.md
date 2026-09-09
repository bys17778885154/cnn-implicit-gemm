# conv5

5 �?3×3 int8 CNN 推理，RTX 4090 Laptop (SM89),仅用 `mma.sync` + `ldmatrix`(�?cp.async)�?

> mma/ldmatrix fragment 布局、K 排序、swizzle、双缓冲流水线的图解详解�?
> **[docs/mma-kernels.md](docs/mma-kernels.md)**(覆盖 m16n8k32 真实版与 m16n16k32 假想对照�?tmp/mma_n16_hypothesis.cu)�?
> Vulkan coopmat 姊妹实现�?`../vulkan`(同模型同验收,四方口径性能对比在两�?README)�?

## 模型

conv(4�?6)+ReLU �?conv(16�?2)+ReLU �?conv(32�?6)+ReLU �?conv(16�?6)+ReLU �?conv(16�?) �?NHWC 输出
输入 1×4×360×796 fp32 NCHW,输出 1×360×796×4 fp32 NHWC(transpose 融合�?conv5 epilogue)�?

## 构建与运�?

```
powershell build.ps1        # CUDA 12.1 + VS2019 工具�? sm_89
build\conv5.exe test        # fragment 微测�?+ 三实现逐层 bit-exact 验证
build\conv5.exe bench       # 计时(3 warmup + 20 次平�?逐层 event 口径)
build\conv5.exe benchcmp    # 四口径计�?GPU �?/ 逐层同步 wall / 异步单链 / 异步×100)
build\conv5.exe dump        # 导出 mma32 最终输�?out_cuda.bin(用于跨实现逐位比对)
build\conv5.exe gen         # 重新生成 weights.bin / input.bin
```

## 实测结果(RTX 4090 Laptop,CC 8.9)

| 实现 | L0 | L1 | L2 | L3 | L4 | total | 有效带宽 |
|---|---|---|---|---|---|---|---|
| v1 naive | 0.902 | 1.798 | 1.787 | 0.906 | 0.231 | 5.625 ms | 9.8 GB/s |
| v2 smem 分块 | 0.571 | 1.132 | 1.191 | 0.575 | 0.177 | 3.646 ms | 15.1 GB/s |
| v3 mma k16 | 0.042 | 0.088 | 0.104 | 0.043 | 0.032 | 0.309 ms | 178.0 GB/s |
| **v3 mma k32** | 0.044 | 0.086 | **0.074** | 0.044 | 0.032 | **0.279 ms** | **197.7 GB/s** |

- `mma.sync.m16n8k16.s8` 版相�?v1 提�?**18×**;`m16n8k32` 版再�?**10%**(总计 20×)
- k32 收益集中�?L2(K=288=9×32 整除,-29%);K=144 的层 = 4 �?k32 �?+ 1 �?k16 残尾�?收益被尾步抵�?±0)
- 三个版本�?CPU int8 参�?*逐层 bit-exact**(整型累加无舍�?验收为精确相�?
- vs CPU fp32 参�?平均绝对误差 0.023(参考输出均�?0.904,平均相对误差 �?2.6%,per-tensor 对称量化的固有噪声水�?

## 跨实现对�?vs Vulkan coopmat,`../vulkan`)

`benchcmp` 模式输出四种口径(GPU 事件�?/ 逐层同步 wall / 异步单链 wall / 异步×100),
�?Vulkan �?*交替背靠�?*测量(同窗�?3 轮取中位):

| 路径 | GPU 吞吐 | 端到�?wall |
|---|---|---|
| **CUDA mma32 逐层 event 同步** | 0.24-0.28 ms | 0.31 ms |
| **CUDA mma32 异步连发**(100 �?1 次同�? | �?| **0.29-0.30 ms** |
| Vulkan coopmat 逐层 submit | 0.50 ms | 0.77 ms |
| Vulkan coopmat 合并 cmdbuffer(100 �?1 次提�? | 0.47 ms | 0.47 ms |

**三方逐位一致�?*:dump 模式导出�?CPU / CUDA / Vulkan 最终输�?SHA256 完全相同
(`0DA656256659892F...`,1,146,240 �?fp32)�?

分析:
- CUDA 全口径领�?**1.6-1.7×**(0.29 vs 0.47ms),即便 Vulkan 已做 command buffer 合并
- CUDA kernel 异步发射开销�?~2-5µs/�?wall 0.30 �?GPU 0.28 + 发射);
  Vulkan 路径每个 dispatch �?~90µs 固定成本(0.47÷5)且与 SM 频率无关
  ——推测为驱动�?compute pipeline 切换/barrier 处理,�?API 层调度开销
- 笔记�?GPU 时钟波动注记:CUDA 偶发 boost 下到�?0.16ms(344 GB/s),�?exe 复测回落 0.28ms;
  跨会话数字不可直接比,跨实现对比一律用交替背靠背口�?

## v3 kernel 要点(src/conv_mma.cu)

- implicit GEMM:M=286560(NPQ) × N=C_out × K=C_in×9;K 排序 `gemm_k = c + C_in*(s+3r)`,每个 k16 �?= 同一 (r,s) �?16 个连续通道 = NHWC �?16B 连续向量
- block tile 128 �?× 全部输出通道;4 warps �?M 划分;`mma.sync.m16n8k16/k32.row.col.s32.s8.s8.s32`
- 权重整块常驻 smem(n 主序,K 连续;K=288 时行�?pad �?304B=19×16B �?ldmatrix �?bank conflict)
- A 双缓�?LDG→寄存器→STS,SM75 风格,�?cp.async):先发下一 k-step �?LDG,再对当前 buffer ldmatrix+mma,�?STS 到另一 buffer
- `ldmatrix.x4` �?A(32 �?,`ldmatrix.x1/x2/x4` �?B(N/8 �?n8 tile);s8 打包:b16 = 2×int8
- k32 变体:A tile 行宽 32B(2×16B chunk),chunk 位置�?`(chunk ^ (row>>2 & 1))` XOR swizzle �?bank conflict;B 用一�?x4 同时�?2 �?n8 tile �?k0-31;K=144 层走 4×k32 + 1×k16 残尾
- epilogue:bias + requant(`__float2int_rn(acc*mult)` clamp[0,127],ReLU 折叠进下�?+ 散写;L4 直接 dequant �?fp32 NHWC
- 资源占用:32-56 寄存�?/ 5-9KB smem / �?spill

## 踩坑记录

1. `ldmatrix` �?warp 集体指令,放入 `if (lane < 8)` 分支会死锁——所�?32 lane 必须执行,仅低 lane 提供有效地址
2. `ldmatrix.xN` 地址取自 lane 0..8N-1(x2 �?lane 0-15),�?lane 地址被忽�?映射错会静默加载重复数据
3. int8 m16n8k16 �?B fragment(n 主序存储)�?*非转�?* ldmatrix 即可,等价�?`*(u32*)&Bs[lane/4][(lane%4)*4]`(�?src/mma_test.cu 微测试实�?
4. CIN=32 时每�?(r,s) �?2 �?k16 �?LDG 需加通道块偏�?`(ks % (CIN/16)) * 16`

## 目录

```
build.ps1        一键构�?
tools/gen_weights.cpp   权重生成 + 校准式量�?模拟 int8 链路逐层�?scale)+ .bin IO
src/quant.h      常量与数据结�?
src/reference.*  CPU fp32 / int8 参�?
src/mma_test.cu  mma/ldmatrix fragment 布局微测�?
src/conv_naive.cu   v1
src/conv_tiled.cu   v2
src/conv_mma.cu     v3
src/main.cpp     验证 + 计时框架
```

