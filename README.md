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
| **Vulkan coopmat n8k32** | 0.213 | 0.260 | 0.474 | 0.213 | 0.192 | 1.352 ms | 41 GB/s |

- **正确性:与 CPU int8 参考 5 层全部 bit-exact**(vs fp32 max_abs=0.15618,与 CUDA 版逐位一致,
  证明 `roundEven` ≡ `lrintf` ≡ `__float2int_rn` 的精度对齐成立)
- 慢 4.8× 的原因(按影响排序):
  1. staging 是标量字节循环(每线程 2×16 次 load/store),CUDA 版是 128b 向量 LDG + swizzle STS
  2. 每个 k-step 两次全 workgroup barrier,无双缓冲(CUDA 版 LDG/mma 重叠)
  3. K=144 pad 到 160,多 1/9 的 mma 与 staging
  4. coopmatLoad/Store 是驱动黑盒(相当于 WMMA 层级),无 ldmatrix 级控制
  5. B 每 step 从 shared 重新 load(CUDA 版常驻 + 复用 fragment 的机会更少)

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
2. GLSL KHR coopmat 命名与 NV 不同:`gl_MatrixUseA/B/Accumulator`(非 MatrixType)、
   `gl_CooperativeMatrixLayoutRowMajor/ColumnMajor`(layout 枚举)、
   `coopMatLoad(m, buf数组, element下标, stride, layout)`——以 KhronosGroup/GLSL 仓库的
   GLSL_KHR_cooperative_matrix.txt 为准
3. coopmat 的 buf 参数要传一维数组(shared 需手动扁平化),element 是起始下标(不是指针)
4. 枚举名是 `VK_COMPONENT_TYPE_SINT8_KHR`(非 SIGNED_INT8)
5. `#extension GL_KHR_cooperative_matrix` 在 SDK 1.4.350 的 glslc 上可用,但常量名要配合
   `GL_KHR_memory_scope_semantics`(gl_ScopeSubgroup 等)
