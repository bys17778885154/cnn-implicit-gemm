# vkconv5

5 层 3×3 int8 CNN 推理,RTX 4090 Laptop (SM89),使用 Vulkan **VK_KHR_cooperative_matrix**(GLSL `coopmat`)。
与 CUDA 版(`../conv5`,mma.sync + ldmatrix)同模型、同 CPU 参考、同 bit-exact 验收标准。

> CUDA 版 mma/ldmatrix fragment 布局的图解详解见 `../conv5/docs/mma-kernels.md`;
> 本项目 kernel 与其对应关系见下方"kernel 要点"。

## 模型

conv(4→16)+ReLU → conv(16→32)+ReLU → conv(32→16)+ReLU → conv(16→16)+ReLU → conv(16→4) → NHWC 输出
输入 1×4×360×796 fp32 NCHW,输出 1×360×796×4 fp32 NHWC(transpose 融合在 conv5 epilogue)。

## 构建与运行

```
powershell build.ps1        # glslc x4 变体(subgroup/workgroup × conv/smoke)+ cl 链接 host
build\vkconv5.exe test      # coopmat 冒烟测试 + 5 层逐层 bit-exact 验证(含属性查询)
build\vkconv5.exe bench     # timestamp 计时(3 warmup + 20 次平均)
```

- 环境:Vulkan SDK 1.4.350(glslc)+ NVIDIA 驱动 596.36 / API 1.4.329;设备枚举显式挑 NVIDIA(跳过 Intel 核显)
- weights.bin / input.bin 与 conv5 共用(复用其 quant.h/reference/gen_weights 源码,首次运行自动生成)
- 运行时查询 coopmat 属性并断言 **M=16 N=8 K=32 s8/s8/s32**(scope=subgroup),同时打印驱动暴露的全部形态
  (含 M=16 N=16 K=32 s8——即 CUDA 版假想文档里的 "m16n16k32",在 coopmat 层真实存在)

## 实测结果(RTX 4090 Laptop)

| 实现 | L0 | L1 | L2 | L3 | L4 | total | 有效带宽 |
|---|---|---|---|---|---|---|---|
| CUDA mma+ldmatrix k16 | 0.042 | 0.088 | 0.104 | 0.043 | 0.032 | 0.309 ms | 178 GB/s |
| CUDA mma+ldmatrix k32 | 0.044 | 0.086 | 0.074 | 0.044 | 0.032 | **0.279 ms** | **198 GB/s** |
| Vulkan coopmat 初版(全标量) | 0.213 | 0.260 | 0.474 | 0.213 | 0.192 | 1.352 ms | 41 GB/s |
| **Vulkan coopmat 优化版** | 0.078 | 0.156 | 0.135 | 0.078 | 0.050 | **0.496 ms** | **111 GB/s** |

- 两版 coopmat 均与 CPU int8 参考**逐层 bit-exact**;vs fp32 max_abs=0.15618,与 CUDA 版逐位一致
  (`roundEven` ≡ `lrintf` ≡ `__float2int_rn` 精度对齐链成立)
- 优化路径(初版 → 优化版 **2.7×**):①staging 向量化(全局 int32×4 加载 + `i8vec4` 打包存储,
  收益 ~2.3×)→ ②寄存器双缓冲(barrier 减半,收益小——mma 每 step 太短藏不住全局延迟)→
  ③block tile BM=192(256 负收益:accs shared 占用翻倍 → 1 block/SM;192 折中最优)
- 剩余 1.78× 差距(CUDA k32 为基准)为 coopmat 抽象层天花板:Load/Store 驱动黑盒
  (无 ldmatrix 级 swizzle/预取控制)、K=144 pad 到 160、B 每 step 重载

## kernel 要点(shaders/conv5_coopmat.comp)

- 与 CUDA mma32 版一一对应:`coopmat<int8,16,32,A>` × `coopmat<int8,32,8,B>` +
  `coopmat<int,16,8,Acc>` ≡ `mma.sync.m16n8k32.s8`;workgroup 192 线程 = 6 subgroups,
  每 subgroup 32 行(M)× C_out(N),2×m16 × NT×n8 个累加器
- **A 路径 = 手动 im2col 中转**(coopmatLoad 只支持连续行 + stride,不支持任意行间接寻址):
  每线程按 `gemm_k = c + C_in*(s+3r)` 排序寻址,4×int32 向量加载 + 位拆包,
  打包成 `i8vec4` 写入 workgroup shared,barrier 后 `coopMatLoad`(row-major, stride 8×4B)
- **B 常驻 shared**(n 主序 `[C_out][K_pad]`,pad 字节清零),`coopMatLoad`(column-major, stride BST)
  ——K=144 统一 pad 到 160(5×32 步,1/9 mma 浪费,单一代码路径;CUDA 版走 k16 残尾步)
- **双缓冲**:下一 k-step 的加载先入 8×int32 寄存器,当前 buffer 上 coopmat 装载 + MulAdd,
  再 STS 到另一 buffer,每步仅 1 次 barrier
- epilogue:`coopMatStore` 累加器 → shared int32 → 逐元素
  `int(roundEven(float(acc+bias_q)*mult))` clamp(0,127)(ReLU 折叠进下界);L4 直写 fp32 NHWC
- 资源占用:As 6KB + Bs ≤9.2KB + accs ≤25KB shared,零 spill
- 双 scope 宏变体(subgroup/workgroup)预编译,按运行时查询结果选择

## 冒烟测试(shaders/smoke.comp)

单条 16×32×8 s8 coopmat vs CPU 精确矩阵乘,bit-exact 才放行——镜像 conv5 的 mma_test.cu 纪律;
且带 `CM_AELEM/CM_BELEM/stride` 探针参数,曾用它实锤驱动 bug(见踩坑 2)。

## 踩坑记录

1. **未初始化的 VkWriteDescriptorSet / VkDescriptorSetLayoutBinding**:pNext / dstArrayElement /
   pImmutableSamplers 是栈上垃圾,驱动解引用直接 AV,且表现为"延迟崩溃"在无关的后续堆分配处
   ——所有 Vulkan 栈上结构体一律 `= {}` 零初始化
2. **驱动 bug(实测确认)**:`coopMatLoad` 对 **vec4 类型 buffer + column-major + 非零 element**
   组合返回错误数据(smoke 探针复现);row-major + vec4 + 非零 element 正常。规避:B 路径用标量
   `int8_t[]`(element/stride 单位=字节)
3. **local_size 与协作循环步长必须单点定义**:装载循环 `i += N` 的 N 与 `local_size_x` 不同步时
   数组只被初始化一半——无报错、结果错,且会污染二分调试(本项目一度把 vec4 优化误判为错误根源)
4. GLSL KHR coopmat 命名与 NV 不同:`gl_MatrixUseA/B/Accumulator`(非 MatrixType)、
   `gl_CooperativeMatrixLayoutRowMajor/ColumnMajor`、`coopMatLoad(m, buf数组, element下标, stride,
   layout)`——以 KhronosGroup/GLSL 仓库的 GLSL_KHR_cooperative_matrix.txt 为准;
   element/stride 单位 = buffer 元素类型(int8 buffer 即字节,i8vec4 buffer 即 4 字节)
5. coopmat 的 buf 参数要传一维数组(shared 需手动扁平化),element 是起始下标(不是指针)
6. 枚举名是 `VK_COMPONENT_TYPE_SINT8_KHR`(非 SIGNED_INT8);`int32_t` 需
   `GL_EXT_shader_explicit_arithmetic_types_int32`;`gl_ScopeSubgroup` 等常量需
   `GL_KHR_memory_scope_semantics`

## 目录

```
build.ps1                    一键构建(glslc x4 + cl)
shaders/conv5_coopmat.comp   主 kernel(手动 im2col + coopmat,spec-constants 参数化 5 层)
shaders/smoke.comp           coopmat 冒烟测试 + 布局/驱动探针
src/main.cpp                 Vulkan host:设备/扩展/属性查询、单 buffer 6 bindings、
                             5 管线 spec-const、逐层下载 memcmp、timestamp 计时
docs/                        spec.md / plan.md
../conv5/                    共用其 quant.h / reference / gen_weights 与 weights.bin
```
