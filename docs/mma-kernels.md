# conv5 mma kernel 详解:m16n8k32(真实)与 m16n16k32(假想)

本文档详细拆解 `src/conv_mma.cu` 中两个 implicit GEMM kernel 的实现:

| kernel | 文件 | 指令 | 状态 |
|---|---|---|---|
| `conv_mma_kernel` | src/conv_mma.cu | `mma.sync.aligned.m16n8k16.s8` | 真实,bit-exact 通过 |
| `conv_mma32_kernel` | src/conv_mma.cu | `mma.sync.aligned.m16n8k32.s8` | 真实,bit-exact 通过 |
| `conv_mma_n16_kernel` | tmp/mma_n16_hypothesis.cu | `mma.sync.aligned.m16n16k32.s8` | **假想**,SM89 不支持,仅作对照 |

以下以 **m16n8k32**(主力)为主线讲解,m16n16k32 只讲差异。

---

## 1. 卷积 → implicit GEMM 的映射

```
卷积 (3×3, stride 1, same padding, NHWC int8)

    x[N=1, H=360, W=796, C_in]      w[C_out, R=3, S=3, C_in]
         │                               │
         ▼                               ▼
   ┌─────────────────────────────────────────────┐
   │  A[M × K]        ×        B[K × N]  = C[M×N]│     M = N·P·Q = 286560 (每输出像素一行)
   │  M = NPQ                  N = C_out         │     K = C_in × 9 (卷积核展开)
   │  K = C_in·R·S             (列主序视角)       │     N 维 = 输出通道
   └─────────────────────────────────────────────┘
         │
         ▼
   y[N, P, Q, C_out]   (NHWC = 矩阵 C 行主序,直接就是最终内存布局)
```

**不做显式 im2col**:A 矩阵的 tile 在 global→shared 加载阶段用下标函数动态生成,
im2col 的像素复制被"寻址时的 (r,s) 偏移 + 越界补零谓词"替代。

### K 维排序(全文最关键的一个决定)

```
gemm_k = c + C_in × (s + 3·r)          (通道 c 放最内层)

一个 k32 步 (gemm_k 连续 32 个) 在两种 C_in 下的几何含义:

C_in = 16 (K=144 = 4×32 + 16 残尾):
  ┌─────────────── k32 步 j ────────────────┐
  │   chunk0 (k 0-15)    │   chunk1 (k16-31)│
  │   (r,s) = (j, j) 展开为 rs=2j           │   rs = s + 3r = 2j
  │   c = 0..15          │   rs = 2j+1      │
  │   → 输入中 16 字节连续 │   c = 0..15      │
  └──────────────────────┴──────────────────┘
  即一个 k32 步 = 卷积核的两个相邻 (r,s) 位置

C_in = 32 (K=288 = 9×32, 无残尾):
  ┌─────────────── k32 步 j ────────────────┐
  │   chunk0: rs = j, c = 0..15             │
  │   chunk1: rs = j, c = 16..31            │   同一 (r,s) 的 32 字节连续!
  └─────────────────────────────────────────┘
```

好处:NHWC 布局下同一像素的 16/32 个通道是**连续字节**,A tile 每个 16B chunk
就是一条 128b 向量加载,不需要任何转置或 shuffle。

---

## 2. 分块层次(Tiling Hierarchy)

```
grid: 2240 个 block (M=286560 / 128,尾块谓词处理)
┌────────────────────────────────────────────────────────────┐
│ block tile:  M=128 行 × N=C_out 全部 (16/32/8)              │
│                                                             │
│  warp0 ─────────────┐                                       │
│  warp1 ─────────────┤  每 warp 负责 32 行(M) × C_out(N)     │
│  warp2 ─────────────┤  = 2 个 m16 tile(M)                   │
│  warp3 ─────────────┘  × NT 个 n8 tile(N), NT = C_out/8    │
│                                                             │
│  smem:  Bs[C_out][K]      权重,常驻(≤9.7KB)                │
│         As[2][128][32]    A 双缓冲,每 k32 步换一块          │
└────────────────────────────────────────────────────────────┘

每 warp 每 k32 步的 mma 条数 = 2(mi) × NT(ni)
     m16n8k32 :  L0/L2/L3(C_out=16): 4 条   L1(32): 8 条   L4(8): 2 条
     m16n16k32:  L0/L2/L3        : 2 条   L1     : 4 条   L4(pad16): 2 条
```

---

## 3. Fragment 布局(m16n8k32,实测验证)

### 3.1 A fragment(4×b32/lane)

A tile = 16 行 × 32B(每行 = 一个输出像素的 32 个 gemm_k 字节):

```
              k 0 ───────── 15 │ 16 ───────── 31
                      chunk0    │     chunk1
  row  0 (像素 m)  ██████████████ │ ██████████████
  row  1           ██████████████ │ ██████████████
   ...
  row  7           ██████████████ │ ██████████████   ◄─ lane 0-7  提供行地址
  row  8           ██████████████ │ ██████████████
   ...
  row 15           ██████████████ │ ██████████████   ◄─ lane 8-15 提供行地址


ldmatrix.x4 的地址映射(一次装满一个 m16×k32 tile):

  lane  0-7  ──►  row 0-7   的 chunk0 (16B)  ──► 寄存器 a0 的矩阵
  lane  8-15 ──►  row 8-15  的 chunk0        ──► 寄存器 a1 的矩阵
  lane 16-23 ──►  row 0-7   的 chunk1        ──► 寄存器 a2 的矩阵
  lane 24-31 ──►  row 8-15  的 chunk1        ──► 寄存器 a3 的矩阵

每 lane 最终持有(b32 = 4 个 s8 打包):
  a0: row = lane/4,    k = 4·(lane%4) .. +3     (k 0-15 段)
  a1: row = lane/4+8,  k 同上
  a2: row = lane/4,    k = 16+4·(lane%4) ..+3   (k 16-31 段)
  a3: row = lane/4+8,  k 同上
```

> 每个 8×8 b16 矩阵被摊到全部 32 个 lane:lane i 拿到该矩阵第 i/4 行、
> 第 4·(i%4) 字节起的 4 个字节——这正是 mma 需要的分布,零 shuffle。

### 3.2 B fragment(2×b32/lane,n8 tile)

B 常驻 smem,**n 主序**(行 = 输出通道,每行 K 字节,gemm_k 连续):

```
Bs[n][k]:          k32 步 ks 的 n8 tile
                     chunk0(k0-15)  chunk1(k16-31)
   n = ni*8+0       ██████████████ ██████████████
   n = ni*8+1       ██████████████ ██████████████
   ...
   n = ni*8+7       ██████████████ ██████████████

一条 ldmatrix.x4 同时装 2 个 n8 tile(C_out=32 时 NT=4,共 2 条 x4):
  lane  0-7  ──►  n(ni0) 0-7 的 chunk0 ──► b[ni0][0]
  lane  8-15 ──►  n(ni0) 0-7 的 chunk1 ──► b[ni0][1]
  lane 16-23 ──►  n(ni1) 0-7 的 chunk0 ──► b[ni1][0]
  lane 24-31 ──►  n(ni1) 0-7 的 chunk1 ──► b[ni1][1]

每 lane 持有:
  b0: n = lane/4, k = 4·(lane%4) .. +3
  b1: n = lane/4, k = 16+4·(lane%4) .. +3
```

> s8 的 B 用**非转置** ldmatrix 即可(与 `*(u32*)&Bs[n][4*(lane%4)]` 等价),
> 由 src/mma_test.cu 微测试实证——int8 fragment 天然按"每 lane 4 个连续 k"分布。

### 3.3 D fragment(4×s32/lane)

```
D tile 16行 × 8列(n8),列 c1 = (lane%4)*2:

            n: c1   c1+1
  row lane/4    d0   d1
  row lane/4+8  d2   d3

epilogue 写回:e=0..3 →  rr = r1 + (e/2)*8,  k = c1 + e%2
```

### 3.4 mma 指令

```
mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
     {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {d0,d1,d2,d3};
      16×8×32 = 4096 MAC/条,累加与输入均为精确整数
```

---

## 4. Shared Memory 布局与 bank conflict 消除

### 4.1 A 缓冲:chunk 级 XOR swizzle(必须)

A 行宽 32B。若 chunk 固定存放,行距 32B = 8 个 4B bank:

```
无 swizzle: row r 与 row r+4 落同一组 bank(相位组内 2-way conflict)
   row0: [chunk0|chunk1]   ─ bank 段 0
   row1: [chunk0|chunk1]   ─ bank 段 2   (32B = 8 bank)
   ...
   row4: [chunk0|chunk1]   ─ bank 段 0  ←与 row0 冲突!

解法:chunk 物理位置 = chunk_idx XOR ((row>>2) & 1)

   row 0-3 :  [c0 | c1]        16B 单元序: 0,1
   row 4-7 :  [c1 | c0]                     1,0   ← 交换
   row 8-11:  [c0 | c1]
   row12-15:  [c1 | c0]

STS 与 ldmatrix 用同一公式计算物理位置 → 每 8-lane 相位组访问
8 个互不相同的 16B bank,零冲突。
```

### 4.2 B 矩阵:行距 padding(仅 K=288 需要 Bs 行距 144B→304B)

```
ldmatrix 相位组内 8 个 lane 读连续 8 行,行距 = BST 字节:
  BST=144 (=9×16B): 16B 单元号 9n mod 8 = n mod 8  → 天然互异 ✓ 无需 pad
  BST=288 (=18×16B): 18n mod 8 = 2n mod 8          → n 与 n+4 冲突 ✗
  pad 到 304 (=19×16B): 19n mod 8 = 3n mod 8       → {0,3,6,1,4,7,2,5} ✓
```

---

## 5. 双缓冲流水线(无 cp.async,SM75 风格)

```
prologue:  LDG(step0) → reg → STS(As[0]) → __syncthreads()

迭代 ks(共 NK32 步 + 可能 1 个 k16 残尾步):
 ┌─────────────────────────────────────────────────────┐
 │ ① LDG(step ks+1) → 寄存器     (尽早发出,延迟窗口)  │
 │ ② ldmatrix(As[ks&1]) → A/B frags                    │
 │ ③ mma ×(2·NT)                 (与 ① 的访存延迟重叠)│
 │ ④ STS(regs → As[(ks+1)&1])    (依赖 ① 完成)        │
 │ ⑤ __syncthreads()                                   │
 └─────────────────────────────────────────────────────┘

时间轴(理想):
 [LDG₁发出][ldmatrix+mma₀ ... 延迟隐藏窗口 ...][STS₁][sync]
           [LDG₂发出][ldmatrix+mma₁ ...........][STS₂][sync] ...

写 As[ks&1] 与读 As[(ks+1)&1] 永远异号缓冲,由 ⑤ 保证无竞争。
```

K=144 的层走 **4 个 k32 步 + 1 个 k16 残尾步**(残尾步用 `m16n8k16` 的旧路径,
A/B fragment 只装 chunk0),K=288 的层走 9 个完整 k32 步。

---

## 6. Epilogue(融合 bias + requant + ReLU + 写回)

```
int32 累加器 (d0..d3)
   │  + bias_q[k]                    (整型加,精确)
   ▼
float v = (float)acc × mult[k]        (mult = s_w·s_a / s_next,离线算好)
   │  __float2int_rn                  (最近偶舍入,与 CPU lrintf bit-exact)
   ▼
clamp(0, 127)                         (下界 0 = ReLU 折叠)
   │
   ▼
中间层: 写 int8 NHWC (y[m·C_out_r + k])
最后一层: mult = s_w·s_a[4],直接写 fp32 NHWC → transpose 天然消失
```

---

## 7. m16n16k32 假想版:与 n8 版的逐项差异

假想指令(N 维翻倍,单条 16×16×32 = 8192 MAC):

```
mma.sync.aligned.m16n16k32.row.col.s32.s8.s8.s32
     {d0..d7}, {a0..a3}, {b0..b3}, {d0..d7}
```

### 7.1 差异对照表

| 项 | m16n8k32(真实) | m16n16k32(假想) |
|---|---|---|
| A fragment | 4×b32,ldmatrix.x4 ×2(每 mi 一条) | **完全相同**(A 与 N 无关) |
| B fragment | 2×b32(n8:1 chunk 对) | **4×b32**:b0=n 0-7 k0-15,b1=n 8-15 k0-15,b2=n 0-7 k16-31,b3=n 8-15 k16-31 |
| B ldmatrix | x4 装 2 个 n8 tile | x4 装 **1 个 n16 tile**(lane:0-7→n0-7 c0,8-15→n8-15 c0,16-23→n0-7 c1,24-31→n8-15 c1) |
| D fragment | 4×s32 | **8×s32**:d4..d7 = n 8-15 半组(k = c1+8 起) |
| 每 warp mma/步 | 2×(C_out/8) | 2×(C_out/16),**减半** |
| NT 定义 | C_out/8(8/16/32 → 1/2/4) | C_out/16(16/32 → 1/2;C_out=8 需 pad 到 16,L4 一半算力空转) |
| 寄存器 | ~56 | 相近(D 总量不变,只是重组) |

### 7.2 D 布局与 epilogue(n16)

```
            n: c1  c1+1      c1+8  c1+9
  row lane/4    d0   d1        d4    d5
  row lane/4+8  d2   d3        d6    d7

写回:e=0..7 →  rr = r1 + ((e/2)%2)*8,  k = c1 + (e/4)*8 + e%2
```

### 7.3 预期性能(见 README 主文档实测背景)

- mma 与 B-ldmatrix 条数减半,但迭代次数(K 步数)不变
- 本网络瓶颈在 LDG 延迟链 + epilogue 写回 + launch(占 ~90%),**预计 0~4%**
- 若与真实 k32 对比单独使用(即 m16n16k16),反而**慢于** k32 版
  (k32 省的是迭代次数/同步次数,n16 只省每步指令数)
- 真正收益场景:C_out ≥ 128 的 compute-bound 卷积,或 SM90 wgmma(N 一次吃 64-256)

---

## 8. 验证方法

1. **fragment 微测试**(src/mma_test.cu):16×16 s8 × 16×8 s8 的单条 mma
   与 CPU 精确矩阵乘对比,穷举 A/B 装载候选(ldmatrix trans/非转置/plain load),
   bit-exact 才放行 → 假想版若上真机,同法验证布局假设
2. **全链 bit-exact**:5 层 GPU 结果与 CPU int8 链路逐层 `memcmp`,
   最终 fp32 输出要求精确相等(整型累加 + 相同 requant 公式)
3. `conv5.exe test / bench`:一键验证 + 计时
