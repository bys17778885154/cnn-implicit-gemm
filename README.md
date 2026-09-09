# vkconv5

conv5(5 层 3×3 int8 CNN)的 Vulkan 实现,使用 **VK_KHR_cooperative_matrix**(GLSL `coopmat`),
与 CUDA 版(`../conv5`,mma.sync + ldmatrix)同模型、同 CPU 参考、同 bit-exact 验收标准。

## 环境

- Vulkan SDK 1.4.350(glslc),NVIDIA 驱动 596.36 / API 1.4.329
- 查询到的 coopmat 属性(节选,scope=subgroup):**M=16 N=8 K=32 A=sint8 B=sint8 C=sint32**
  (驱动还暴露 M=16 N=16 K=32 s8 形态——即 CUDA 版假想文档里的 "n16k32" 在 coopmat 层面是真实存在的,
  只是我们按设计仍用 n8 形态)
- 设备枚举显式挑 NVIDIA(跳过 Intel 核显)

## 构建 / 运行

```
powershell build.ps1        # glslc x4 变体(subgroup/workgroup × conv/smoke)+ cl 链接
build\vkconv5.exe test      # coopmat 冒烟测试 + 5 层逐层 bit-exact 验证
build\vkconv5.exe bench     # timestamp 计时(3 warmup + 20 平均)
```

weights.bin / input.bin 与 conv5 共用(首次运行自动从 conv5 的同源代码生成)。

## 结果(RTX 4090 Laptop,SM89)

| 实现 | L0 | L1 | L2 | L3 | L4 | total | 有效带宽 |
|---|---|---|---|---|---|---|---|
| CUDA mma+ldmatrix k16 | 0.042 | 0.088 | 0.104 | 0.043 | 0.032 | 0.309 ms | 178 GB/s |
| CUDA mma+ldmatrix k32 | 0.044 | 0.086 | 0.074 | 0.044 | 0.032 | **0.279 ms** | **198 GB/s** |
| Vulkan coopmat 初版(全标量) | 0.213 | 0.260 | 0.474 | 0.213 | 0.192 | 1.352 ms | 41 GB/s |
| **Vulkan coopmat 优化版** | 0.078 | 0.156 | 0.135 | 0.078 | 0.050 | **0.496 ms** | **111 GB/s** |

优化项(全部保持 bit-exact):
1. **staging 向量化**:全局加载 int32×4 + 位拆包,shared 存储打包 `i8vec4`(8 次 vec4 store 替代 32 次字节 store)——收益最大(~2.3×)
2. **寄存器双缓冲**:下一 k-step 的加载先入寄存器,与当前 mma 重叠,barrier 减半——收益小(mma 太短,藏不住全局延迟)
3. **block tile BM=192**:256 负收益(accs 占用翻倍 → 1 block/SM),192 折中最优(~5%)

- 最终与 CUDA k32 差距 **1.78×**(初版 4.8×);剩余差距来自:coopmatLoad/Store 驱动黑盒、K pad 到 160、B 每 step 重载
- **正确性:两版均与 CPU int8 参考 5 层全部 bit-exact**(vs fp32 max_abs=0.15618,与 CUDA 版逐位一致,
  证明 `roundEven` ≡ `lrintf` ≡ `__float2int_rn` 的精度对齐成立)

## 结构

```
shaders/conv5_coopmat.comp   主 kernel:手动 im2col → workgroup shared → coopmatLoad/MulAdd/Store
shaders/smoke.comp           冒烟:单条 16×32×8 s8 coopmat vs CPU 精确矩阵乘
src/main.cpp                 Vulkan host(设备/扩展/属性查询、单 buffer 6 bindings、5 管线 spec-const 参数化、
                              逐层下载 memcmp、timestamp 计时),复用 ../conv5 的 quant.h/reference
```

kernel 与 CUDA mma32 版的对应关系:
`coopmat<int8,16,32,A>` × `coopmat<int8,32,8,B>` + `coopmat<int,16,8,Acc>` ≡ `mma.sync.m16n8k32.s8`;
A staging 的 gemm_k = c + C_in*(s+3r) 排序、B 的 n 主序常驻、epilogue 的
`int(roundEven(float(acc+bq)*mult))` clamp(0,127) 全部与 CUDA 版同构。

## 踩坑记录

1. **未初始化的 VkWriteDescriptorSet / VkDescriptorSetLayoutBinding 结构体**:pNext /
   dstArrayElement / pImmutableSamplers 是栈上垃圾,驱动解引用直接 AV(表现为"延迟崩溃"在
   无关的后续堆分配处)——所有 Vulkan 栈上结构体一律 `= {}` 零初始化
2. **驱动 bug(实测确认)**:`coopMatLoad` 对 **vec4 类型 buffer + column-major + 非零 element**
   的组合返回错误数据(专用 smoke 探针复现);row-major + vec4 + 非零 element 正常。
   规避:B 路径用标量 `int8_t[]`(element 单位=字节)
3. **local_size 与协作循环步长必须一致**:权重装载/epilogue 循环 `i += N` 的 N 与
   `local_size_x` 不同步时,数组只被初始化一半——症状是结果错但无任何报错,且会污染
   二分调试(本项目一度把 vec4 优化误判为错误根源)
4. GLSL KHR coopmat 命名与 NV 不同:`gl_MatrixUseA/B/Accumulator`(非 MatrixType)、
   `gl_CooperativeMatrixLayoutRowMajor/ColumnMajor`(layout 枚举)、
   `coopMatLoad(m, buf数组, element下标, stride, layout)`——以 KhronosGroup/GLSL 仓库的
   GLSL_KHR_cooperative_matrix.txt 为准;element/stride 单位 = buffer 元素类型
5. coopmat 的 buf 参数要传一维数组(shared 需手动扁平化),element 是起始下标(不是指针)
6. 枚举名是 `VK_COMPONENT_TYPE_SINT8_KHR`(非 SIGNED_INT8)
7. `#extension GL_KHR_cooperative_matrix` 在 SDK 1.4.350 的 glslc 上可用,但常量名要配合
   `GL_KHR_memory_scope_semantics`(gl_ScopeSubgroup 等);`int32_t` 需
   `GL_EXT_shader_explicit_arithmetic_types_int32`
