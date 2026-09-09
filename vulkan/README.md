# vkconv5

5 层 3×3 int8 CNN 推理,RTX 4090 Laptop (SM89),使用 Vulkan **VK_KHR_cooperative_matrix**(GLSL `coopmat`)。
与 CUDA 版(`../cuda`,mma.sync + ldmatrix)同模型、同 CPU 参考、同 bit-exact 验收标准。

> CUDA 版 mma/ldmatrix fragment 布局的图解详解见 `../cuda/docs/mma-kernels.md`;
> 本项目 kernel 与其对应关系见下方"kernel 要点"。

## 模型

conv(4→16)+ReLU → conv(16→32)+ReLU → conv(32→16)+ReLU → conv(16→16)+ReLU → conv(16→4) → NHWC 输出
输入 1×4×360×796 fp32 NCHW,输出 1×360×796×4 fp32 NHWC(transpose 融合在 conv5 epilogue)。

## 构建与运行

```
powershell build.ps1        # glslc x6 变体(subgroup/workgroup × n8/n16/smoke)+ cl 链接 host
build\vkconv5.exe test      # 冒烟测试 + n8/n16 双版本逐层 bit-exact 验证(含属性查询)
build\vkconv5.exe bench     # 四口径并列(n8 与 n16 自动对比)
build\vkconv5.exe benchmany # 合并稳态:100 链一条提交(n8 与 n16)
build\vkconv5.exe dump      # 导出 out_vk.bin + out_cpu.bin(跨实现逐位比对,n8)
```

- 环境:Vulkan SDK 1.4.350(glslc)+ NVIDIA 驱动 596.36 / API 1.4.329;设备枚举显式挑 NVIDIA(跳过 Intel 核显)
- weights.bin / input.bin 与 CUDA 版共用同一生成逻辑(`../common/gen_weights.cpp`,确定性生成,逐位一致)
- 运行时查询 coopmat 属性并断言 **M=16 N=8 K=32 s8/s8/s32**(scope=subgroup),同时打印驱动暴露的全部形态

## 实测结果(RTX 4090 Laptop,交替背靠背同窗口测量)

| 路径 | GPU 吞吐 | 端到端 wall |
|---|---|---|
| CUDA mma32 逐层 event 同步 | 0.24-0.28 ms | 0.31 ms |
| CUDA mma32 异步连发(100 链/同步一次) | — | **0.29-0.30 ms** |
| Vulkan coopmat 逐层 submit | 0.42 ms | 0.65 ms |
| Vulkan coopmat 合并 cmdbuffer(100 链/提交) | 0.40-0.43 ms | **0.40-0.43 ms** |

**三方逐位一致性已验证**:`dump` 模式导出 CPU / CUDA(mma32)/ Vulkan(合并链)的最终输出,
SHA256 完全相同(`0DA656256659892F...`,1,146,240 个 fp32)。

结论:
- CUDA 全口径领先约 1.5×;合并 cmdbuffer 让 Vulkan 端到端快 ~35%,但每 dispatch ~80µs
  固定成本(CUDA kernel 发射仅 ~2-5µs)是 API 层调度差距,详见顶层 README 性能汇总
- 笔记本 GPU 时钟波动注记:跨会话数字不可直接比,跨实现对比一律交替背靠背口径
- bench 尾部自动做完整性校验:下载最终输出与 CPU 参考精确比对,通过才输出成绩

## N=16 形态支持与实测(coopmat 的 "m16n16k32")

驱动属性转储中存在 **M=16 N=16 K=32 s8/s8/s32**(scope=subgroup),shader 通过 `USE_N16`
宏支持(`coopmat<int8,32,16,B>` + `coopmat<int,16,16,Acc>`;B 装载保持标量 int8
column-major 规避已知驱动 bug;L4 的 C_out=8 pad 到 16,新增 spec-const `CG` 区分全局权重
行宽与 shader COUT)。test/bench/benchmany 自动双版本对比:

| 形态 | L0 | L1 | L2 | L3 | L4 | benchmany | bit-exact |
|---|---|---|---|---|---|---|---|
| N=8(32×8 B) | 0.065 | 0.113 | 0.123 | 0.065 | 0.050 | **0.426 ms** | ✅ |
| N=16(32×16 B) | 0.065 | 0.113 | 0.124 | 0.065 | 0.061 | 0.433 ms(-1.8%) | ✅ |

实证了 `../cuda/docs/mma-kernels.md` 假想分析的两个预测:
- MulAdd 指令数减半**几乎不可见**——本 kernel 延迟/dispatch 受限(tensor 利用率 ~10%),
  不是 mma issue 受限;n16 只省每步指令数,不减迭代次数与 barrier
- L4(C_out=4→pad 16)按预测明显负收益(0.050→0.061ms,+22%):一半 tile 空转
- 该形态适合 C_out ≥ 128 的 compute-bound 卷积,不适合本网络

## kernel 要点(shaders/conv5_coopmat.comp)

- 与 CUDA mma32 版一一对应:`coopmat<int8,16,32,A>` × `coopmat<int8,32,8|16,B>` +
  `coopmat<int,16,8|16,Acc>` ≡ `mma.sync.m16n8k32.s8`(n16 形态见上节);workgroup 192 线程
  = 6 subgroups,每 subgroup 32 行(M)× C_out(N)
- **A 路径 = 手动 im2col 中转**(coopmatLoad 只支持连续行 + stride):每线程按
  `gemm_k = c + C_in*(s+3r)` 排序寻址,4×int32 向量加载 + 位拆包,打包 `i8vec4` 写入
  workgroup shared,barrier 后 `coopMatLoad`(row-major, stride 8×4B)
- **B 常驻 shared**(n 主序 `[C_out][K_pad]`,pad 清零),`coopMatLoad`(column-major, stride BST);
  K=144 统一 pad 到 160(单一代码路径;CUDA 版走 k16 残尾步)
- **双缓冲**:下一 k-step 加载先入 8×int32 寄存器,当前 buffer 上装载 + MulAdd,再 STS 到
  另一 buffer,每步仅 1 次 barrier
- epilogue:按 subgroup 分 6 轮,`coopMatStore` 到单 subgroup 整行 scratch `[32][COUT]`
  (保持 16/32B 连续写回),全 192 线程 requant
  `int(roundEven(float(acc+bias_q)*mult))` clamp(0,127);L4 直写 fp32 NHWC
- **shared 峰值 21KB(L1 层)**,满足 40KB 约束;演进数据见 git 历史
  (整块 accs 41KB / 16×8 tile 碎片写负优化 / 现行整行 scratch +11% 提速)
- 双 scope 宏(subgroup/workgroup)+ N=8/16 宏共 4 个 conv spv 变体预编译

## 冒烟测试(shaders/smoke.comp)

单条 16×32×8 s8 coopmat vs CPU 精确矩阵乘,bit-exact 才放行——镜像 CUDA 版 mma_test.cu 的纪律;
带 `CM_AELEM/CM_BELEM/stride` 探针参数,曾用它实锤驱动 bug(见踩坑 2)。

## 踩坑记录

1. **未初始化的 VkWriteDescriptorSet / VkDescriptorSetLayoutBinding**:栈上垃圾被驱动解引用
   直接 AV,表现为延迟崩溃——Vulkan 栈上结构体一律 `= {}` 零初始化
2. **驱动 bug(实测确认)**:`coopMatLoad` 对 **vec4 类型 buffer + column-major + 非零 element**
   返回错误数据(smoke 探针复现);row-major + vec4 正常。规避:B 路径用标量 `int8_t[]`
3. **驱动 bug(实测确认)**:合并 cmdbuffer 内**多组 timestamp 在 fence 完成后不可用**;
   规避:整链 wall 计时(QPC),per-layer 单层提交计时
4. **`vkAllocateDescriptorSets` 批量分配 `pSetLayouts` 必须指向数组**:count=5 传单个指针
   → set 损坏,录制时静默崩溃
5. **"空 command buffer" 测量陷阱**:`l <= -1` 循环零执行 → 空 cb 提交被误读为 6.7× 提速;
   `all` 模式残留结果又骗过完整性校验。修正:显式循环边界 + benchmany 均值 + 校验即时显式打印
6. **local_size 与协作循环步长必须单点定义**:不同步 → 数组半初始化,静默出错
7. GLSL KHR coopmat 命名:`gl_MatrixUseA/B/Accumulator`、`gl_CooperativeMatrixLayoutRowMajor/
   ColumnMajor`、`coopMatLoad(m, buf数组, element下标, stride, layout)`;element/stride 单位 =
   buffer 元素类型
8. coopmat 的 buf 参数传一维数组(shared 手动扁平化),element 是起始下标
9. 枚举名 `VK_COMPONENT_TYPE_SINT8_KHR`;`int32_t` 需 int32 扩展;`gl_ScopeSubgroup` 需
   `GL_KHR_memory_scope_semantics`

## 目录

```
build.ps1                    一键构建(glslc x6 + cl)
shaders/conv5_coopmat.comp   主 kernel(USE_N16/SCOPE 宏变体,spec-constants 参数化)
shaders/smoke.comp           coopmat 冒烟测试 + 布局/驱动探针
src/main.cpp                 host:属性查询、双 pipeline 集(n8/n16)、合并提交、四口径计时
../common/                   共享层:quant.h / reference.* / gen_weights.cpp(与 CUDA 版共用)
```
