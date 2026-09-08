# vkconv5 实现计划

> 内联执行(同会话)。完整代码落在 vkconv5/ 下;本计划锁定任务边界与验收命令。
> 规格见 docs/spec.md。复用:../conv5/src/quant.h, reference.*, tools/gen_weights.cpp;同一份 weights.bin/input.bin。

## 任务

1. **骨架**:git init、.gitignore、build.ps1(glslc ×2 变体 + cl 链接 host,VC2019 + VulkanSDK 1.4.350)
   - 验收:空 shader 能编译出 spv
2. **shader**:`shaders/conv5_coopmat.comp`
   - spec-const: 0 CIN / 1 COUT(=pad) / 2 COUT_R / 3 LAST / 4 KS / 5 KP / 6 BST
   - 常量:W=796,H=360,HW,block=128
   - 结构:权重装载(含 pad 清零)→ im2col staging(k32 步循环:staging→barrier→coopmatLoad A/B→MultiplyAdd)→ acc store → requant 写回
   - 双 scope 宏变体
   - 验收:glslc 编译通过(函数名/语法按报错迭代)
3. **host V0**:instance/设备枚举(挑 NVIDIA)/扩展与特性启用/CooperativeMatrixPropertiesKHR 查询与断言
   - 验收:运行打印 16/8/32 + scope
4. **V1 冒烟**:`shaders/smoke.comp`(local_size=32,A 16×32、B 32×8 从 storage buffer 读,一条 MultiplyAdd,store int32)+ host 复用大 buffer 与提交框架
   - 验收:随机数据 vs CPU 乘 bit-exact
5. **V2 全链**:descriptor 按层更新、5 pipeline、dispatch 2240 workgroups、逐层下载 memcmp、timestamp 计时(3 warmup + 20 平均)
   - 验收:test 全层 PASS;bench 输出 L0..L4 ms + total + GB/s,格式对齐 CUDA 版
6. **README**:对比表(coopmat vs CUDA mma/mma32),坑点记录,commit

## 验收命令
```
powershell build.ps1
build\vkconv5.exe test
build\vkconv5.exe bench
```
weights.bin/input.bin 从 ../conv5 复制或 gen(调用 conv5 逻辑同源代码)。
