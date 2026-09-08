# vkconv5 — VK_KHR_cooperative_matrix 版 5 层 int8 CNN 设计

日期:2026-09-08 | 状态:已获用户认可
目标:用 Vulkan `VK_KHR_cooperative_matrix`(GLSL coopmat)在 RTX 4090 Laptop 实现与 conv5(CUDA mma32)同标准的 5 层 int8 卷积推理,逐层 bit-exact + 性能对比。

## 环境(已验证)
- Vulkan SDK 1.4.350(glslc/glslangValidator),NVIDIA 驱动 596.36 / API 1.4.329
- VK_KHR_cooperative_matrix rev2 已暴露;系统另有 Intel 核显 → 必须显式选 NVIDIA 设备

## 架构
- 姊妹项目 `D:\b00852572\vkconv5`,`#include ../conv5/src/quant.h|reference.h`,共用 weights.bin/input.bin、CPU fp32/int8 参考、bit-exact 验收
- 映射:coopmat<int8,16,32,A> × coopmat<int8,32,8,B> + coopmat<int,16,8,acc> ≡ CUDA m16n8k32;运行时查询 s8/s8/s32 属性断言 M=16,N=8,K=32,记录 scope(subgroup/workgroup),选对应预编译 spv(两个宏变体)
- Shader(compute, local_size=128 = 4 subgroups):
  - 每 subgroup 32 行(M)× C_out(N),2×m16 × NT×n8 个 acc coopmat
  - A 路径 = 手动 im2col:每线程按 gemm_k = c + CIN*(s+3r) 寻址,写 workgroup shared `As[128][32]`(越界/gk≥KS 补零),barrier 后 cooperativeMatrixLoad(row-major, stride 32)
  - B 常驻 shared `Bs[COUT][BST]`(n 主序;K=144 层 pad 到 160,pad 字节清零),cooperativeMatrixLoad(column-major, stride BST)
  - K=144 → 统一 pad 到 160(5×32 步,1/9 mma 浪费,单一代码路径;与 CUDA 版 k16 残尾不同,记录在案)
  - Epilogue:cooperativeMatrixStore → shared int32 [128][COUT] → barrier → 每元素 `int(roundEven(float(acc+bq)*mult))` clamp(0,127) / L4 直写 fp32
  - 精度对齐:roundEven(最近偶) ≡ lrintf ≡ __float2int_rn;OpFMul ≡ mulss ≡ __fmul_rn → bit-exact
- Host:Vulkan 1.3 设备,启用 cooperativeMatrix + shaderInt8 + storageBuffer8BitAccess + vulkanMemoryModel 特性;单个大 device buffer(23MB),6 个 typed bindings(输入/输出int8/权重/bias/mult/最终fp32)按层更新 descriptor;5 个 pipeline(spec-constants: CIN/COUT/COUT_R/LAST/KS/KP/BST);timestamp query 计时
- 不做双缓冲(v1 留优化),不用 BDA/VMA

## 里程碑
| 阶段 | 验收 |
|---|---|
| V0 设备/属性查询 | 打印 M/N/K/scope = 16/8/32/subgroup |
| V1 coopmat 冒烟测试(16×32×8 单条) | vs CPU bit-exact |
| V2 全链 5 层 + 逐层下载比对 + 计时 | 与 CPU int8 逐层 bit-exact;输出与 CUDA 同格式对比表 |

## 风险与对策
- GLSL 扩展/内建函数名(GL_KHR_cooperative_matrix 的 load/store/multiplyAdd 精确拼写)→ 构建期试错修正
- scope 模板参数须常量 → 双变体 spv(-DSCOPE_SUBGROUP 1/0),按查询结果选择
- int8 storage → 启 8bit storage 特性;shared 数组 spec-constant 尺寸(合法)
