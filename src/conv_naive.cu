#include "gpu.h"
#include <cuda_runtime.h>

template <int CIN, int COUT, int COUT_R, bool LAST>
__global__ void conv_naive_kernel(const int8_t* __restrict__ a, void* __restrict__ yv,
                                  const int8_t* __restrict__ B, const int32_t* __restrict__ bq,
                                  const float* __restrict__ mult) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = HW * COUT_R;
    if (idx >= total) return;
    int k = idx % COUT_R;
    int m = idx / COUT_R;
    int q = m % W;
    int p = m / W;
    int acc = 0;
#pragma unroll
    for (int rs = 0; rs < 9; ++rs) {
        int r = rs / 3, s = rs % 3;
        int h = p + r - 1, w = q + s - 1;
        if (h < 0 || h >= H || w < 0 || w >= W) continue;
        const int8_t* ar = a + ((size_t)h * W + w) * CIN;
        const int8_t* br = B + (size_t)rs * CIN * COUT + k;
#pragma unroll
        for (int c = 0; c < CIN; ++c) acc += (int)ar[c] * (int)br[(size_t)c * COUT];
    }
    acc += bq[k];
    if (LAST) {
        ((float*)yv)[(size_t)m * COUT_R + k] = __fmul_rn((float)acc, mult[k]);
    } else {
        int v = __float2int_rn(__fmul_rn((float)acc, mult[k]));
        v = max(0, min(v, 127));
        ((int8_t*)yv)[(size_t)m * COUT_R + k] = (int8_t)v;
    }
}

static void launch_naive(const int8_t* a, void* y, const int8_t* B, const int32_t* bq,
                         const float* mult, int l) {
    int total = HW * C_OUT_REAL[l];
    dim3 g((total + 255) / 256), t(256);
    switch (l) {
    case 0: conv_naive_kernel<16, 16, 16, false><<<g, t>>>(a, y, B, bq, mult); break;
    case 1: conv_naive_kernel<16, 32, 32, false><<<g, t>>>(a, y, B, bq, mult); break;
    case 2: conv_naive_kernel<32, 16, 16, false><<<g, t>>>(a, y, B, bq, mult); break;
    case 3: conv_naive_kernel<16, 16, 16, false><<<g, t>>>(a, y, B, bq, mult); break;
    case 4: conv_naive_kernel<16, 8, 4, true><<<g, t>>>(a, y, B, bq, mult); break;
    }
}

void conv5_naive_chain(const GpuModel& g, float* layer_ms, int last_layer) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    void* bufs[6] = { (void*)g.a0, g.x[0], g.x[1], g.x[0], g.x[1], g.out };
    int lastn = last_layer < 0 ? 4 : last_layer;
    for (int l = 0; l <= lastn; ++l) {
        cudaGetLastError();
        cudaEventRecord(e0);
        launch_naive((const int8_t*)bufs[l], bufs[l + 1], g.b[l], g.bq[l], g.mult[l], l);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) printf("naive launch L%d error: %s\n", l, cudaGetErrorString(err));
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&layer_ms[l], e0, e1);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}
