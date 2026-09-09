# cnn-int8

5 层 3×3 int8 CNN(输入 1×4×360×796)在 RTX 4090 Laptop (SM89) 上的两种 GPU 实现,
单仓库、双实现、零耦合:两种实现互不引用,只共享 `common/` 的模型层
(量化/CPU 参考/权重生成),保证 bit-exact 验收用同一把尺子。

## 结构

```
common/            共享层:quant.h(常量/模型)、reference.cpp(CPU fp32+int8 参考)、
                   gen_weights.cpp(校准式量化 + .bin IO)——两个实现唯一的公共依赖
cuda/              CUDA 实现(mma.sync + ldmatrix,无 cp.async),独立构建运行 → conv5.exe
vulkan/            Vulkan 实现(VK_KHR_cooperative_matrix),独立构建运行 → vkconv5.exe
```

## 独立构建与验证(各自目录内完成,互不依赖)

```
cd cuda;    powershell build.ps1; .\build\conv5.exe   test / bench / benchcmp / dump
cd vulkan;  powershell build.ps1; .\build\vkconv5.exe test / bench / benchmany / dump
```

工具链互不相交:cuda = nvcc(sm_89),vulkan = glslc + cl(Vulkan SDK)。
weights.bin/input.bin 各目录各持一份(确定性生成,内容逐位一致)。

## 跨实现结论(详见两个子 README)

- **三方逐位一致**:CPU / CUDA / Vulkan 最终输出 SHA256 相同(`0DA656256659892F...`)
- 四口径性能对比(交替背靠背):CUDA mma32 异步连发 **0.29-0.30ms**,
  Vulkan coopmat 合并 cmdbuffer **0.40-0.43ms**(峰值 shared 21KB,满足 40KB 约束);
  CUDA 全口径领先约 1.5×(Vulkan 每 dispatch ~80µs 固定成本 vs CUDA kernel 发射 ~2-5µs)

## 历史

本仓库由 conv5 与 vkconv5 两个仓库经 `git subtree` 合并而成,双方全部提交历史保留
(`git log --follow common/quant.h` 可追溯至原仓库)。
