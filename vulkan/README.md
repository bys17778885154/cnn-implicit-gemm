# vkconv5

5 �?3×3 int8 CNN 推理,RTX 4090 Laptop (SM89),使用 Vulkan **VK_KHR_cooperative_matrix**(GLSL `coopmat`)�?
�?CUDA �?`../common �� ../cuda`,mma.sync + ldmatrix)同模型、同 CPU 参考、同 bit-exact 验收标准�?

> CUDA �?mma/ldmatrix fragment 布局的图解详解见 `../common �� ../cuda/docs/mma-kernels.md`;
> 本项�?kernel 与其对应关系见下�?kernel 要点"�?

## 模型

conv(4�?6)+ReLU �?conv(16�?2)+ReLU �?conv(32�?6)+ReLU �?conv(16�?6)+ReLU �?conv(16�?) �?NHWC 输出
输入 1×4×360×796 fp32 NCHW,输出 1×360×796×4 fp32 NHWC(transpose 融合�?conv5 epilogue)�?

## 构建与运�?

```
powershell build.ps1        # glslc x4 变体(subgroup/workgroup × conv/smoke)+ cl 链接 host
build\vkconv5.exe test      # coopmat 冒烟测试 + 5 层逐层 bit-exact 验证(含属性查�?
build\vkconv5.exe bench     # 四口径并�?GPU 时间戳和 / 逐层 submit wall / 合并 wall
build\vkconv5.exe benchmany # 合并稳�?100 链一条提�?取每链均�?
build\vkconv5.exe dump      # 导出 out_vk.bin + out_cpu.bin(跨实现逐位比对)
```

- 环境:Vulkan SDK 1.4.350(glslc)+ NVIDIA 驱动 596.36 / API 1.4.329;设备枚举显式�?NVIDIA(跳过 Intel 核显)
- weights.bin / input.bin �?conv5 共用(复用�?quant.h/reference/gen_weights 源码,首次运行自动生成)
- 运行时查�?coopmat 属性并断言 **M=16 N=8 K=32 s8/s8/s32**(scope=subgroup),同时打印驱动暴露的全部形�?
  (�?M=16 N=16 K=32 s8——即 CUDA 版假想文档里�?"m16n16k32",�?coopmat 层真实存�?

## 实测结果(RTX 4090 Laptop,交替背靠背同窗口测量)

**四方口径对比**(每格 20 次平�?交替运行 3 轮取中位):

| 路径 | GPU 吞吐 | 端到�?wall |
|---|---|---|
| CUDA mma32 逐层 event 同步 | 0.24-0.28 ms | 0.31 ms |
| CUDA mma32 异步连发(100 �?同步一�? | �?| **0.29-0.30 ms** |
| Vulkan coopmat 逐层 submit | 0.50 ms | 0.77 ms |
| Vulkan coopmat 合并 cmdbuffer(100 �?提交) | 0.47 ms | **0.47 ms** |

**三方逐位一致性已验证**:`dump` 模式导出 CPU / CUDA(mma32)/ Vulkan(合并�?的最终输�?
SHA256 完全相同(`0DA656256659892F...`,1,146,240 �?fp32)�?

结论:
- **CUDA 全口径领�?1.6-1.7×**(0.29 vs 0.47ms)。合�?cmdbuffer �?Vulkan 端到端快 30%
  (0.77�?.47ms,5 �?host 提交往返变 1 �?,但追不平 CUDA:CUDA kernel 异步发射开销�?
  ~2-5µs/�?�?Vulkan 路径每个 dispatch �?~90µs 的固定成�?0.47ms ÷ 5,推测来自
  驱动�?compute pipeline 切换/barrier 处理,�?SM 频率无关)
- 笔记�?GPU 时钟波动注记:CUDA 偶发 boost 状态下可达 0.16ms(344 GB/s),�?exe 复测回落
  0.28ms;跨会话数字不可直接比,本表为交替背靠背口径
- coopmat 版与 CUDA 的剩余差距构�?上述 dispatch 固定成本 + coopmatLoad/Store 黑盒 +
  K pad �?160 + B �?step 重载
- bench 尾部自动做完整性校�?下载最终输出与 CPU 参考精确比�?通过才输出成�?

## kernel 要点(shaders/conv5_coopmat.comp)

- �?CUDA mma32 版一一对应:`coopmat<int8,16,32,A>` × `coopmat<int8,32,8,B>` +
  `coopmat<int,16,8,Acc>` �?`mma.sync.m16n8k32.s8`;workgroup 192 线程 = 6 subgroups,
  �?subgroup 32 �?M)× C_out(N),2×m16 × NT×n8 个累加器
- **A 路径 = 手动 im2col 中转**(coopmatLoad 只支持连续行 + stride,不支持任意行间接寻址):
  每线程按 `gemm_k = c + C_in*(s+3r)` 排序寻址,4×int32 向量加载 + 位拆�?
  打包�?`i8vec4` 写入 workgroup shared,barrier �?`coopMatLoad`(row-major, stride 8×4B)
- **B 常驻 shared**(n 主序 `[C_out][K_pad]`,pad 字节清零),`coopMatLoad`(column-major, stride BST)
  ——K=144 统一 pad �?160(5×32 �?1/9 mma 浪费,单一代码路径;CUDA 版走 k16 残尾�?
- **双缓�?*:下一 k-step 的加载先�?8×int32 寄存�?当前 buffer �?coopmat 装载 + MulAdd,
  �?STS 到另一 buffer,每步�?1 �?barrier
- epilogue:`coopMatStore` 累加�?�?shared int32 �?逐元�?
  `int(roundEven(float(acc+bias_q)*mult))` clamp(0,127)(ReLU 折叠进下�?;L4 直写 fp32 NHWC
- 资源占用:As 6KB + Bs �?.2KB + accs �?5KB shared,�?spill
- �?scope 宏变�?subgroup/workgroup)预编�?按运行时查询结果选择

## 冒烟测试(shaders/smoke.comp)

单条 16×32×8 s8 coopmat vs CPU 精确矩阵�?bit-exact 才放行——镜�?conv5 �?mma_test.cu 纪律;
且带 `CM_AELEM/CM_BELEM/stride` 探针参数,曾用它实锤驱�?bug(见踩�?2)�?

## 踩坑记录

1. **未初始化�?VkWriteDescriptorSet / VkDescriptorSetLayoutBinding**:pNext / dstArrayElement /
   pImmutableSamplers 是栈上垃�?驱动解引用直�?AV,且表现为"延迟崩溃"在无关的后续堆分配处
   ——所�?Vulkan 栈上结构体一�?`= {}` 零初始化
2. **驱动 bug(实测确认)**:`coopMatLoad` �?**vec4 类型 buffer + column-major + 非零 element**
   组合返回错误数据(smoke 探针复现);row-major + vec4 + 非零 element 正常。规�?B 路径用标�?
   `int8_t[]`(element/stride 单位=字节)
3. **驱动 bug(实测确认)**:合并 command buffer 内的**多组 timestamp �?fence 完成后仍不可�?*
   (`vkGetQueryPoolResults`+WAIT 挂死或返�?0),单提交单�?timestamp 正常。规�?整链只测
   wall(QPC),per-layer 计时用单层独立提�?
4. **`vkAllocateDescriptorSets` 批量分配�?`pSetLayouts` 必须指向数组**:count=5 却传单个 layout
   指针,驱动越界读栈垃圾,分配"成功"�?set 已损�?后续录制静默崩溃
5. **"�?command buffer" 测量陷阱(本项目最大教�?**:`run_chain(-1)` �?`l <= -1` 使录制循�?
   一次都不执�?�?�?cmdbuffer 提交,fence 几十微秒即完�?**被误读为 6.7× 提�?*;更隐蔽的�?
   `all` 模式�?bench 的赛后校验下载到的是 **validate 阶段残留的正确结�?*,假象双重自洽�?
   修正:①计数循环用显式边界;②计时基准用 benchmany(100 链一条提交取均�?排除单次假象);
   ③完整性校验必须在"该进程内该路径确实执行过"之后立即�?且成功要显式打印
6. **local_size 与协作循环步长必须单点定�?*:装载循环 `i += N` �?N �?`local_size_x` 不同步时
   数组只被初始化一半——无报错、结果错,且会污染二分调试
6. GLSL KHR coopmat 命名�?NV 不同:`gl_MatrixUseA/B/Accumulator`(�?MatrixType)�?
   `gl_CooperativeMatrixLayoutRowMajor/ColumnMajor`、`coopMatLoad(m, buf数组, element下标, stride,
   layout)`——以 KhronosGroup/GLSL 仓库�?GLSL_KHR_cooperative_matrix.txt 为准;
   element/stride 单位 = buffer 元素类型(int8 buffer 即字�?i8vec4 buffer �?4 字节)
7. coopmat �?buf 参数要传一维数�?shared 需手动扁平�?,element 是起始下�?不是指针)
8. 枚举名是 `VK_COMPONENT_TYPE_SINT8_KHR`(�?SIGNED_INT8);`int32_t` 需
   `GL_EXT_shader_explicit_arithmetic_types_int32`;`gl_ScopeSubgroup` 等常量需
   `GL_KHR_memory_scope_semantics`

## 目录

```
build.ps1                    一键构�?glslc x4 + cl)
shaders/conv5_coopmat.comp   �?kernel(手动 im2col + coopmat,spec-constants 参数�?5 �?
shaders/smoke.comp           coopmat 冒烟测试 + 布局/驱动探针
src/main.cpp                 Vulkan host:设备/扩展/属性查询、单 buffer 6 bindings�?
                             5 管线 spec-const、逐层下载 memcmp、timestamp 计时
docs/                        spec.md / plan.md
../common �� ../cuda/                    共用�?quant.h / reference / gen_weights �?weights.bin
```

