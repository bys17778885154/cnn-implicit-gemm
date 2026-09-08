# 5 层 int8 CNN — CUDA mma + ldmatrix 实现设计文档

日期:2026-09-08
状态:已获用户认可
目标:自写 CUDA kernel 练手,在 RTX 4090 Laptop (SM89) 上用 `mma.sync` + `ldmatrix`(不使用 cp.async)实现 5 层 3×3 卷积网络的 int8 推理

## 1. 模型定义

| 层 | 输入 | 输出 | 权重 (KCRS) | bias | 激活 |
|---|---|---|---|---|---|
| conv1 | 1×4×360×796 | 1×16×360×796 | 16×4×3×3 | 16 | ReLU |
| conv2 | 1×16×360×796 | 1×32×360×796 | 32×16×3×3 | 32 | ReLU |
| conv3 | 1×32×360×796 | 1×16×360×796 | 16×16×3×3 | 16 | ReLU |
| conv4 | 1×16×360×796 | 1×16×360×796 | 16×16×3×3 | 16 | ReLU |
| conv5 | 1×16×360×796 | 1×4×360×796 | 4×16×3×3 | 4 | 无 |

- 全部 3×3、stride 1、same padding(H、W 不变)
- 输入 1×4×360×796(fp32, NCHW),输出 1×360×796×4(fp32, NHWC = NCHW 经 transpose)
- 最终 transpose 通过 conv5 直接输出 NHWC 实现,无独立转置 kernel

## 2. 数据流与量化

- 内部激活布局:NHWC int8。输入经预处理 kernel 量化并转 NHWC;conv1 的 4 通道 zero-pad 到 16(K 从 36 变 144,统一 k16 对齐)
- K 维排序:`gemm_k = c + C_in * (s + S * r)`(通道最内层)。每个 k16 块 = 一个 (r,s) 的 16 个连续通道 = NHWC 中 16B 连续内存,A tile 加载天然 128b 向量
- 权重离线重排:`W[K_out][C_in][R][S] (KCRS)` → `B[gemm_k][K_out]` int8,存 .bin
- 量化方案(对称):
  - 权重 per-output-channel:`s_w[k] = max|W[k,:,:,:]| / 127`
  - 激活 per-tensor:`s_a = max|A| / 127`
  - 中间层 epilogue:bias(int32)+ requant(`round(acc * s_a * s_w[k] / s_next)` fp32 乘)+ clamp[-128,127] + ReLU,保持 int8
  - conv5 epilogue:bias + fp32 dequant,直接写 NHWC fp32
  - 累加安全:|Σ| ≤ 127×127×144 ≈ 2.3e6 << 2^31
- 验收精度:GPU int8 输出与 CPU int8 参考 **bit-exact**(整数运算无舍入、顺序无关);另报告 vs CPU fp32 的相对误差

## 3. Kernel 设计

### GEMM 视图(每层)
`C[M=NPQ=286560, N=C_out] = A[M, K=C_in*9] × B[K, N]`,N∈{16,32,16,16,4(pad到8)},K∈{144}(conv1 pad 后统一)

### v3 终态(implicit GEMM + mma + ldmatrix)
- block tile = M128 × N全量;4 warps 沿 M 划分,每 warp 32 行 × N 列
- `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32`;每 warp 每 k-step 2~8 条 mma
- B(权重)整块常驻 smem(≤144×32 int8 = 4.5KB)
- A tile 每 k-step 128×16 int8 = 2KB;128 线程每人一次 128b LDG
- `ldmatrix.sync.aligned.m8n8.x4.shared.b16`:2 个 int8 打包为 1 个 b16,一条装满一个 fragment
- 128b 粒度 XOR swizzle:`store_col = (row % 8) ^ col16B_idx`,T0..T7 无 bank conflict
- 双缓冲(无 cp.async,SM75 风格):k-step 内先发下一块 A 的 LDG 到寄存器 → 当前 buffer ldmatrix+mma → STS(swizzle)到另一 buffer → `__syncthreads()`
- Epilogue:acc 经 smem 重排 [M,N] 行主序 → 中间层 requant+ReLU 向量写 int8;conv5 dequant 写 fp32
- 边界:M 尾块(286560 % 128 = 96)谓词;A 加载 p/q 越界补零

### v1 / v2(前置里程碑)
- v1:naive conv,每线程一个输出像素,直接卷积循环,验证量化链路与 bit-exact 框架
- v2:NHWC + smem 分块 + 128b 向量化加载,不含 mma

## 4. 工程结构与构建

```
conv5/
├── docs/specs/2026-09-08-int8-cnn-mma-design.md   # 本文档
├── tools/gen_weights.cpp   # 随机权重 → 量化 → 重排 B → weights.bin(含 bias/scale)
├── src/
│   ├── main.cpp            # 加载 .bin、逐层调用、cudaEvent 计时、校验
│   ├── reference.cpp       # CPU fp32 卷积 + CPU int8 模拟参考
│   ├── quant.h             # 量化参数结构、requant 常量
│   ├── nchw_to_nhwc.cu     # 预处理(fp32 NCHW → int8 NHWC, pad C=4→16)
│   ├── conv_naive.cu       # v1
│   ├── conv_tiled.cu       # v2
│   └── conv_mma.cu         # v3,模板 <C_in, C_out> 实例化 5 层
├── build.ps1               # vcvars64(VS2019) + nvcc -arch=sm_89
└── README.md
```

- 构建环境(已在本机验证):CUDA 12.1 + VS2019 MSVC 14.29 toolset(cl 14.44 过新被 CUDA 12.1 拒绝),`-arch=sm_89`
- 权重/输入由 `tools/gen_weights.cpp` 生成为 .bin,运行时由 main 加载
- 计时:cudaEvent,预热 3 次 + 平均 20 次,报告 per-layer ms 与总 ms、有效带宽 GB/s

## 5. 里程碑与验收

| 版本 | 内容 | 验收标准 |
|---|---|---|
| v0 | CPU 参考 + 权重工具 | .bin 生成,CPU fp32/int8 参考 baseline |
| v1 | GPU naive conv | 与 CPU int8 参考 bit-exact;vs fp32 相对误差 < 2% |
| v2 | smem 分块向量化 | 同上;性能 ≥ v1 |
| v3 | mma + ldmatrix | 同上;报告性能,分析带宽利用率 |

每版完成后运行验收,失败则修复后才进入下一版。

## 6. 明确排除项(YAGNI)

- 不做 cp.async / TMA / wgmma(用户明确要求;TMA/wgmma 本也不支持 SM89)
- 不做多 batch(batch 固定 1)
- 不做 Winograd、int8 定点 multiplier-shift 优化(requant 用 fp32 乘,如性能不达标再议)
- 不做训练/反向
