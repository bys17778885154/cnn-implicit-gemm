#include "../src/gpu.h"
#include <cuda_runtime.h>

template <int CIN, int COUT, int COUT_R, bool LAST>
__global__ void conv_mma_n16_kernel(const int8_t* __restrict__ a, void* __restrict__ yv,
                                    const int8_t* __restrict__ B, const int32_t* __restrict__ bq,
                                    const float* __restrict__ mult) {
    constexpr int KS = CIN * 9;
    constexpr int BST = KS + ((KS / 16) % 2 == 0 ? 16 : 0);
    constexpr int NK32 = KS / 32;
    constexpr int TAIL = (KS % 32) != 0 ? 1 : 0;
    constexpr int NSTEP = NK32 + TAIL;
    constexpr int NT = COUT / 16;

    __shared__ int8_t Bs[COUT][BST];
    __shared__ __align__(16) int8_t As[2][128][32];

    int tid = threadIdx.x;
    for (int i = tid; i < COUT * KS; i += 128) {
        int gk = i % KS;
        int k = i / KS;
        Bs[k][gk] = B[(size_t)gk * COUT + k];
    }

    int row0 = blockIdx.x * 128 + tid;
    int p0 = row0 / W, q0 = row0 % W;
    {
        int4 v0 = make_int4(0, 0, 0, 0), v1 = make_int4(0, 0, 0, 0);
        if (row0 < HW) {
            int h = p0 - 1, w = q0 - 1;
            if (h >= 0 && w >= 0) v0 = *(const int4*)(a + ((size_t)h * W + w) * CIN);
            int rs = 16 / CIN;
            h = p0 + rs / 3 - 1;
            w = q0 + rs % 3 - 1;
            if (h >= 0 && h < H && w >= 0 && w < W)
                v1 = *(const int4*)(a + ((size_t)h * W + w) * CIN + 16 % CIN);
        }
        *(int4*)&As[0][tid][((tid >> 2) & 1) * 16] = v0;
        *(int4*)&As[0][tid][(((tid >> 2) & 1) ^ 1) * 16] = v1;
    }
    __syncthreads();

    int warp = tid / 32, lane = tid % 32;
    int wbase = warp * 32;
    int d[NT][2][8];
#pragma unroll
    for (int ni = 0; ni < NT; ++ni)
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int j = 0; j < 8; ++j) d[ni][mi][j] = 0;

#pragma unroll 1
    for (int ks = 0; ks < NSTEP; ++ks) {
        int4 n0 = make_int4(0, 0, 0, 0), n1 = make_int4(0, 0, 0, 0);
        bool store = false;
        if (ks + 1 < NSTEP) {
            store = true;
            int gk0 = (ks + 1) * 32;
            if (row0 < HW) {
                int rs = gk0 / CIN, c0 = gk0 % CIN;
                int h = p0 + rs / 3 - 1, w = q0 + rs % 3 - 1;
                if (h >= 0 && h < H && w >= 0 && w < W)
                    n0 = *(const int4*)(a + ((size_t)h * W + w) * CIN + c0);
                if (ks + 1 < NK32) {
                    int rs2 = (gk0 + 16) / CIN, c2 = (gk0 + 16) % CIN;
                    int h2 = p0 + rs2 / 3 - 1, w2 = q0 + rs2 % 3 - 1;
                    if (h2 >= 0 && h2 < H && w2 >= 0 && w2 < W)
                        n1 = *(const int4*)(a + ((size_t)h2 * W + w2) * CIN + c2);
                }
            }
        }

        int buf = ks & 1;
        if (ks < NK32) {
            uint32_t af[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                int rabs = wbase + mi * 16 + (lane % 8) + ((lane >> 3) & 1) * 8;
                int phys = (((lane >> 4) & 1) ^ ((rabs >> 2) & 1)) * 16;
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(&As[buf][rabs][phys]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(af[mi][0]), "=r"(af[mi][1]), "=r"(af[mi][2]), "=r"(af[mi][3]) : "r"(addr));
            }
            uint32_t b[NT][4];
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                int nrow = ni * 16 + (lane % 8) + ((lane >> 3) & 1) * 8;
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(&Bs[nrow][ks * 32 + ((lane >> 4) & 1) * 16]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b[ni][0]), "=r"(b[ni][1]), "=r"(b[ni][2]), "=r"(b[ni][3]) : "r"(addr));
            }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                for (int ni = 0; ni < NT; ++ni)
                    asm volatile("mma.sync.aligned.m16n16k32.row.col.s32.s8.s8.s32 "
                                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
                                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                                 : "+r"(d[ni][mi][0]), "+r"(d[ni][mi][1]), "+r"(d[ni][mi][2]), "+r"(d[ni][mi][3]),
                                   "+r"(d[ni][mi][4]), "+r"(d[ni][mi][5]), "+r"(d[ni][mi][6]), "+r"(d[ni][mi][7])
                                 : "r"(af[mi][0]), "r"(af[mi][1]), "r"(af[mi][2]), "r"(af[mi][3]),
                                   "r"(b[ni][0]), "r"(b[ni][1]), "r"(b[ni][2]), "r"(b[ni][3]));
        } else {
            uint32_t a0, a1, a2, a3;
            {
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(&As[buf][wbase + lane][((lane >> 2) & 1) * 16]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
            }
            uint32_t b[NT][2];
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(&Bs[ni * 16 + lane % 16][NK32 * 32]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                             : "=r"(b[ni][0]), "=r"(b[ni][1]) : "r"(addr));
            }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                for (int ni = 0; ni < NT; ++ni)
                    asm volatile("mma.sync.aligned.m16n16k16.row.col.s32.s8.s8.s32 "
                                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                                 : "+r"(d[ni][mi][0]), "+r"(d[ni][mi][1]), "+r"(d[ni][mi][2]), "+r"(d[ni][mi][3]),
                                   "+r"(d[ni][mi][4]), "+r"(d[ni][mi][5]), "+r"(d[ni][mi][6]), "+r"(d[ni][mi][7])
                                 : "r"(mi == 0 ? a0 : a2), "r"(mi == 0 ? a1 : a3),
                                   "r"(b[ni][0]), "r"(b[ni][1]));
        }
        if (store) {
            *(int4*)&As[(ks + 1) & 1][tid][((tid >> 2) & 1) * 16] = n0;
            *(int4*)&As[(ks + 1) & 1][tid][(((tid >> 2) & 1) ^ 1) * 16] = n1;
        }
        __syncthreads();
    }

    int mbase = blockIdx.x * 128 + warp * 32;
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            int r1 = mbase + mi * 16 + lane / 4;
            int c1 = ni * 16 + (lane % 4) * 2;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                int rr = r1 + ((e / 2) % 2) * 8;
                int k = c1 + (e / 4) * 8 + e % 2;
                if (rr >= HW || k >= COUT_R) continue;
                int v = d[ni][mi][e] + bq[k];
                if (LAST) {
                    ((float*)yv)[(size_t)rr * COUT_R + k] = __fmul_rn((float)v, mult[k]);
                } else {
                    int y = __float2int_rn(__fmul_rn((float)v, mult[k]));
                    y = max(0, min(y, 127));
                    ((int8_t*)yv)[(size_t)rr * COUT_R + k] = (int8_t)y;
                }
            }
        }
    }
}

static void launch_n16(const int8_t* a, void* y, const int8_t* B, const int32_t* bq,
                       const float* mult, int l) {
    dim3 g((HW + 127) / 128), t(128);
    switch (l) {
    case 0: conv_mma_n16_kernel<16, 16, 16, false><<<g, t>>>(a, y, B, bq, mult); break;
    case 1: conv_mma_n16_kernel<16, 32, 32, false><<<g, t>>>(a, y, B, bq, mult); break;
    case 2: conv_mma_n16_kernel<32, 16, 16, false><<<g, t>>>(a, y, B, bq, mult); break;
    case 3: conv_mma_n16_kernel<16, 16, 16, false><<<g, t>>>(a, y, B, bq, mult); break;
    case 4: conv_mma_n16_kernel<16, 16, 4, true><<<g, t>>>(a, y, B, bq, mult); break;
    }
}

void conv5_mma_n16_chain(const GpuModel& g, float* layer_ms, int last_layer) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    void* bufs[6] = { (void*)g.a0, g.x[0], g.x[1], g.x[0], g.x[1], g.out };
    int lastn = last_layer < 0 ? 4 : last_layer;
    for (int l = 0; l <= lastn; ++l) {
        cudaGetLastError();
        cudaEventRecord(e0);
        launch_n16((const int8_t*)bufs[l], bufs[l + 1], g.b[l], g.bq[l], g.mult[l], l);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) printf("mma_n16 launch L%d error: %s\n", l, cudaGetErrorString(err));
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&layer_ms[l], e0, e1);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}
