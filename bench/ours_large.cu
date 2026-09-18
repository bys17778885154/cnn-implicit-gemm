#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include "conv_kernel.cuh"

static const int H = 360, W = 796, HW = H * W;

static int32_t cpu_ref(const std::vector<int8_t>& a, const std::vector<int8_t>& b,
                       int C, int K, int row, int k) {
    int p = row / W, q = row % W;
    int32_t acc = 0;
    for (int r = 0; r < 3; ++r) {
        int h = p + r - 1;
        if (h < 0 || h >= H) continue;
        for (int s = 0; s < 3; ++s) {
            int w = q + s - 1;
            if (w < 0 || w >= W) continue;
            const int8_t* ar = &a[((size_t)h * W + w) * C];
            const int8_t* br = &b[(((size_t)k * 3 + r) * 3 + s) * C];
            for (int c = 0; c < C; ++c) acc += (int)ar[c] * (int)br[c];
        }
    }
    return acc;
}

static uint32_t lcg = 12345;
static int8_t rnd8() { lcg = lcg * 1664525u + 1013904223u; return (int8_t)(lcg >> 24); }

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    constexpr int C = 64, K = 64;
    std::vector<int8_t> a((size_t)HW * C), b((size_t)K * 9 * C);
    for (auto& v : a) v = rnd8();
    for (auto& v : b) v = rnd8();
    std::vector<int8_t> bg((size_t)K * 9 * C);
    for (int k = 0; k < K; ++k)
        for (int rs = 0; rs < 9; ++rs)
            for (int c = 0; c < C; ++c)
                bg[(size_t)(c + C * rs) * K + k] = b[(((size_t)k * 9 + rs) * C) + c];

    int8_t* dA; int8_t* dB; int32_t* dC;
    CUDA_CHECK(cudaMalloc(&dA, a.size()));
    CUDA_CHECK(cudaMalloc(&dB, bg.size()));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)HW * K * 4));
    CUDA_CHECK(cudaMemcpy(dA, a.data(), a.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, bg.data(), bg.size(), cudaMemcpyHostToDevice));

    dim3 g((HW + 127) / 128), t(128);
    for (int i = 0; i < 3; ++i) conv_mma32_kernel<C, K, K, false, true><<<g, t>>>(dA, dC, dB, nullptr, nullptr, dC);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < 20; ++i) conv_mma32_kernel<C, K, K, false, true><<<g, t>>>(dA, dC, dB, nullptr, nullptr, dC);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    ms /= 20;
    cudaEventDestroy(e0); cudaEventDestroy(e1);

    std::vector<int32_t> host((size_t)HW * K);
    CUDA_CHECK(cudaMemcpy(host.data(), dC, host.size() * 4, cudaMemcpyDeviceToHost));
    uint32_t seed = 999;
    bool ok = true;
    for (int tt = 0; tt < 16 && ok; ++tt) {
        seed = seed * 1664525u + 1013904223u;
        int row = (seed >> 8) % HW;
        seed = seed * 1664525u + 1013904223u;
        int k = (seed >> 8) % K;
        if (host[(size_t)row * K + k] != cpu_ref(a, b, C, K, row, k)) {
            printf("[FAIL] row=%d k=%d got %d want %d\n", row, k,
                   host[(size_t)row * K + k], cpu_ref(a, b, C, K, row, k));
            ok = false;
        }
    }
    double gops = 2.0 * HW * K * 9.0 * C;
    double bytes = (double)HW * (C + 4.0 * K);
    printf("ours    mma32 64->64: %.3f ms  %.0f GOP/s  %.1f GB/s  tensor-util=%.0f%%  spot-check %s\n",
           ms, gops / ms / 1e6, bytes / ms / 1e6, gops / ms / 1e6 / 226000.0 * 100.0,
           ok ? "OK" : "FAIL");
    return 0;
}



