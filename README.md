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

| 实现 | 耗时 | 有效带宽 | 验证 |
|---|---|---|---|
| CUDA mma+ldmatrix k16(逐层 launch + event) | 0.309 ms | 178 GB/s | bit-exact |
| CUDA mma+ldmatrix k32(逐层 launch + event) | 0.279 ms | 198 GB/s | bit-exact |
| Vulkan coopmat 逐层 dispatch(初版全标量) | 1.352 ms | 41 GB/s | bit-exact |
| Vulkan coopmat 逐层 dispatch(kernel 优化后) | 0.496 ms | 111 GB/s | bit-exact |
| Vulkan coopmat 单 command buffer 合并(benchmany:100 链/提交) | 0.477 ms | 115 GB/s | bit-exact + 赛后完整性校验 |

**三方逐位一致性已验证**:`dump` 模式导出 CPU / CUDA(mma32)/ Vulkan(合并链)的最终输出,
SHA256 完全相同(`0DA656256659892F...`,1,146,240 个 fp32)。

- command buffer 合并的真实收益仅 **~4%**(0.496 → 0.477ms),此前宣称的 6.7× 是测量 bug
  (见踩坑 5)。单次提交的 wall(0.557ms)甚至略慢——host 侧录制/提交开销与 GPU 时间重叠不足
- coopmat 版与 CUDA k32 的 1.7× 差距仍在:coopmatLoad/Store 驱动黑盒、K pad 到 160、
  B 每 step 重载——WMMA 级 API 的固有天花板
- bench 尾部自动做完整性校验:下载最终输出与 CPU 参考精确比对,通过才输出成绩

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
3. **驱动 bug(实测确认)**:合并 command buffer 内的**多组 timestamp 在 fence 完成后仍不可用**
   (`vkGetQueryPoolResults`+WAIT 挂死或返回 0),单提交单组 timestamp 正常。规避:整链只测
   wall(QPC),per-layer 计时用单层独立提交
4. **`vkAllocateDescriptorSets` 批量分配时 `pSetLayouts` 必须指向数组**:count=5 却传单个 layout
   指针,驱动越界读栈垃圾,分配"成功"但 set 已损坏,后续录制静默崩溃
5. **"空 command buffer" 测量陷阱(本项目最大教训)**:`run_chain(-1)` 中 `l <= -1` 使录制循环
   一次都不执行 → 空 cmdbuffer 提交,fence 几十微秒即完成,**被误读为 6.7× 提速**;更隐蔽的是
   `all` 模式下 bench 的赛后校验下载到的是 **validate 阶段残留的正确结果**,假象双重自洽。
   修正:①计数循环用显式边界;②计时基准用 benchmany(100 链一条提交取均值,排除单次假象);
   ③完整性校验必须在"该进程内该路径确实执行过"之后立即做,且成功要显式打印
6. **local_size 与协作循环步长必须单点定义**:装载循环 `i += N` 的 N 与 `local_size_x` 不同步时
   数组只被初始化一半——无报错、结果错,且会污染二分调试
6. GLSL KHR coopmat 命名与 NV 不同:`gl_MatrixUseA/B/Accumulator`(非 MatrixType)、
   `gl_CooperativeMatrixLayoutRowMajor/ColumnMajor`、`coopMatLoad(m, buf数组, element下标, stride,
   layout)`——以 KhronosGroup/GLSL 仓库的 GLSL_KHR_cooperative_matrix.txt 为准;
   element/stride 单位 = buffer 元素类型(int8 buffer 即字节,i8vec4 buffer 即 4 字节)
7. coopmat 的 buf 参数要传一维数组(shared 需手动扁平化),element 是起始下标(不是指针)
8. 枚举名是 `VK_COMPONENT_TYPE_SINT8_KHR`(非 SIGNED_INT8);`int32_t` 需
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
