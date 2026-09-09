# Tensor Core 利用率实测(ncu 计数器 + nsys 采样,RTX 4090 Laptop)

测量日期:2026-09-09;两种独立方法互为印证。

## 方法一:Nsight Compute 硬件计数器(CUDA mma32,`dump` 模式逐层)

| 层 | Tensor 管线利用率 | IMMA 指令数 | kernel 时长* | SM SOL | Memory SOL |
|---|---|---|---|---|---|
| L0 (16→16) | 17.2% | 179,120 | 54.0 µs | 57.0% | 72.2% |
| L1 (16→32) | 16.5% | 358,240 | 108.5 µs | 47.9% | 75.2% |
| L2 (32→16) | 19.8% | 322,416 | 93.2 µs | 50.9% | 84.2% |
| L3 (16→16) | 17.2% | 179,120 | 54.1 µs | 56.9% | 72.1% |
| L4 (16→4)  | 12.2% | 89,560 | 38.7 µs | 54.0% | 68.0% |

\* ncu 重放含插桩开销(bench 口径约 43µs 的 L0 在此为 54µs),利用率按周期计不受影响。

**IMMA 指令数与代码结构逐层精确吻合**(计数器可信性验证):
每 warp 指令数 = L0/L3: 20(4×k32+1×k16 残尾 × 2mi × 2ni)、L1: 40(NT=4)、L2: 36(9×k32 步)、
L4: 10;× 2240 blocks × 4 warps = 8,960 warps,与表中总数一致。

## 方法二:Nsight Systems GPU 指标采样(10kHz,两侧同会话同时钟 1455 MHz)

| 方案 | 负载 | GPU 忙时 Tensor Active | GPU 忙时 SM Active |
|---|---|---|---|
| CUDA mma32 | benchcmp(含异步×100) | **15.0%** | 91.1% |
| Vulkan coopmat | benchmany(100 链合并提交) | **9.9%** | 95.9% |

与 ncu 逐层聚合(~15%)一致 ✓;CUDA:Vulkan = 1.5×,恰等于性能比(0.29 vs 0.43ms)。

## 结论

1. **两个实现的 Tensor Core 利用率都很低(10-20%)**——本工作负载是访存/延迟受限,
   不是算力受限;CUDA 的 Memory SOL 高达 68-84%,才是真正的瓶颈线
2. SM Active 高达 91-96% 但 Tensor 管线空闲 → SM 在跑 staging/requant/barrier 等非 mma
   工作(与"CUDA 领先 1.5× 来自调度与访存模式,而非 mma 本身"的既有结论闭环)
3. 想提高 Tensor 利用率的路子 = 提高算术强度:更大 C_out 的模型、或感受野分块融合
   (中间激活不落显存)——与本仓库 README 中"融合方向"分析一致

## 工具踩坑

- Nsight Compute 2023.1(随机驱动 596.36)抓不到 Vulkan kernel(`No kernels were profiled`
  但 app 正常运行)→ Vulkan 侧改用 nsys GPU 指标采样
- nsys 2023.1.2 无 `gpu_metric_util` 报表;GPU 指标以 JSON 形式存于导出 sqlite 的
  `GENERIC_EVENTS` 表(来源 GpuMetrics),自行解析统计
- nsys `stats` 直接生成的 sqlite 会截损(文件头为空),须用 `nsys export --type sqlite` 重导

## 复现

```
ncu --csv -k "regex:conv_mma32" --launch-count 5 --log-file ncu_cuda.csv .\build\conv5.exe dump
nsys profile --gpu-metrics-device=0 --gpu-metrics-frequency=10000 -o vk_trace .\build\vkconv5.exe benchmany
nsys export --type sqlite -o vk_trace.sqlite vk_trace.nsys-rep   # 再解析 GENERIC_EVENTS
```
