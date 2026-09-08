#include "gpu.h"
#include <cuda_runtime.h>

template <int CIN, int COUT, int COUT_R, bool LAST>
__global__ void conv_tiled_kernel(const int8_t* __restrict__ a, void* __restrict__ yv,
                                  const int8_t* __restrict__ B, const int32_t* __restrict__ bq,
                                  const float* __restrict__ mult) {
    __shared__ int8_t As[10][18][CIN];
    __shared__ int8_t Ws[COUT][CIN * 9];
    int tx = threadIdx.x;
    int q0 = blockIdx.x * 16, p0 = blockIdx.y * 8;
    int tq = tx % 16, tp = tx / 16;

#pragma unroll
    for (int i = tx; i < 10 * 18 * (CIN / 16); i += 128) {
        int wq = i % 18;
        int hp = (i / 18) % 10;
        int c0 = i / 180;
        int h = p0 - 1 + hp, w = q0 - 1 + wq;
        int4 v = make_int4(0, 0, 0, 0);
        if (h >= 0 && h < H && w >= 0 && w < W)
            v = *(const int4*)(a + ((size_t)h * W + w) * CIN + c0 * 16);
        *(int4*)&As[hp][wq][c0 * 16] = v;
    }
#pragma unroll
    for (int i = tx; i < COUT * CIN * 9; i += 128) {
        int gk = i % (CIN * 9);
        int k = i / (CIN * 9);
        Ws[k][gk] = B[(size_t)gk * COUT + k];
    }
    __syncthreads();

    int acc[COUT];
#pragma unroll
    for (int k = 0; k < COUT; ++k) acc[k] = 0;
    int p = p0 + tp, q = q0 + tq;
#pragma unroll
    for (int rs = 0; rs < 9; ++rs) {
        int r = rs / 3, s = rs % 3;
#pragma unroll
        for (int c = 0; c < CIN; ++c) {
            int av = (int)As[tp + r][tq + s][c];
#pragma unroll
            for (int k = 0; k < COUT; ++k)
                acc[k] += av * (int)Ws[k][c + CIN * rs];
        }
    }
    if (p < H && q < W) {
#pragma unroll
        for (int k = 0; k < COUT_R; ++k) {
            int v = acc[k] + bq[k];
            if (LAST) {
                ((float*)yv)[((size_t)p * W + q) * COUT_R + k] = __fmul_rn((float)v, mult[k]);
            } else {
                int y = __float2int_rn(__fmul_rn((float)v, mult[k]));
                y = max(0, min(y, 127));
                ((int8_t*)yv)[((size_t)p * W + q) * COUT_R + k] = (int8_t)y;
            }
        }
    }
}

static void launch_tiled(const int8_t* a, void* y, const int8_t* B, const int32_t* bq,
                         const float* mult, int l) {
    dim3 g((W + 15) / 16, (H + 7) / 8), t(128);
    switch (l) {
    case 0: conv_tiled_kernel<16, 16, 16, false><<<g, t>>>(a, y, B, bq, mult); break;
    case 1: conv_tiled_kernel<16, 32, 32, false><<<g, t>>>(a, y, B, bq, mult); break;
    case 2: conv_tiled_kernel<32, 16, 16, false><<<g, t>>>(a, y, B, bq, mult); break;
    case 3: conv_tiled_kernel<16, 16, 16, false><<<g, t>>>(a, y, B, bq, mult); break;
    case 4: conv_tiled_kernel<16, 8, 4, true><<<g, t>>>(a, y, B, bq, mult); break;
    }
}

void conv5_tiled_chain(const GpuModel& g, float* layer_ms, int last_layer) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    void* bufs[6] = { (void*)g.a0, g.x[0], g.x[1], g.x[0], g.x[1], g.out };
    int lastn = last_layer < 0 ? 4 : last_layer;
    for (int l = 0; l <= lastn; ++l) {
        cudaGetLastError();
        cudaEventRecord(e0);
        launch_tiled((const int8_t*)bufs[l], bufs[l + 1], g.b[l], g.bq[l], g.mult[l], l);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) printf("tiled launch L%d error: %s\n", l, cudaGetErrorString(err));
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&layer_ms[l], e0, e1);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}
