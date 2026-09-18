#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static const int H = 360, W = 796, HW = H * W;

template <int TB_N, int WB_N, int STAGES>
struct Cfg {
    using Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        int8_t, cutlass::layout::TensorNHWC,
        int8_t, cutlass::layout::TensorNHWC,
        int32_t, cutlass::layout::TensorNHWC,
        int32_t,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, TB_N, 64>,
        cutlass::gemm::GemmShape<64, WB_N, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombination<
            int32_t, 4, int32_t, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        STAGES,
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::conv::IteratorAlgorithm::kAnalytic>::Kernel;
    using Op = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
};

template <typename Op>
static double run_case(int C, int K, int8_t* dA, int8_t* dB, int32_t* dC,
                       std::vector<int32_t>& host) {
    using LayoutT = cutlass::layout::TensorNHWC;
    typename Op::Arguments args{
        cutlass::conv::Conv2dProblemSize(1, H, W, C, K, 3, 3, H, W, 1, 1, 1, 1, 1, 1,
                                         cutlass::conv::Mode::kCrossCorrelation),
        {(int8_t*)dA, LayoutT(LayoutT::packed({1, H, W, C}))},
        {(int8_t*)dB, LayoutT(LayoutT::packed({K, 3, 3, C}))},
        {dC, LayoutT(LayoutT::packed({1, H, W, K}))},
        {dC, LayoutT(LayoutT::packed({1, H, W, K}))},
        {1.0f, 0.0f}};
    Op op;
    if (op.can_implement(args) != cutlass::Status::kSuccess) { printf("cannot implement\n"); exit(1); }
    if (op.initialize(args, nullptr) != cutlass::Status::kSuccess) { printf("init fail\n"); exit(1); }
    for (int i = 0; i < 3; ++i) op.run(nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < 20; ++i) op.run(nullptr);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    ms /= 20;
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    host.assign((size_t)HW * K, 0);
    CUDA_CHECK(cudaMemcpy(host.data(), dC, host.size() * 4, cudaMemcpyDeviceToHost));
    return ms;
}

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

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int maxC = 256;
    int8_t* dA; int8_t* dB; int32_t* dC;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)HW * maxC));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)maxC * 9 * maxC));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)HW * maxC * 4));

    struct Case { int C, K; const char* tag; };
    Case cases[] = { {64, 64, "64->64"}, {128, 128, "128->128"}, {256, 256, "256->256"} };

    for (auto& cs : cases) {
        int C = cs.C, K = cs.K;
        std::vector<int8_t> a((size_t)HW * C), b((size_t)K * 9 * C);
        for (auto& v : a) v = rnd8();
        for (auto& v : b) v = rnd8();
        CUDA_CHECK(cudaMemcpy(dA, a.data(), a.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, b.data(), b.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dC, 0, (size_t)HW * K * 4));

        double gops = 2.0 * HW * K * 9.0 * C;
        double bytes = (double)HW * (C + 4.0 * K);
        std::vector<int32_t> host;
        double ms = run_case<Cfg<128, 64, 3>::Op>(C, K, dA, dB, dC, host);
        double gops_per_s = gops / (ms * 1e-3);
        printf("cutlass 128x128x64s3 %s: %.3f ms  %.0f GOP/s  %.1f GB/s  tensor-util=%.0f%%\n",
               cs.tag, ms, gops_per_s / 1e9, bytes / ms / 1e6,
               gops_per_s / 1e9 / 226.0 * 100.0);
        uint32_t seed = 999;
        bool ok = true;
        for (int t = 0; t < 16 && ok; ++t) {
            seed = seed * 1664525u + 1013904223u;
            int row = (seed >> 8) % HW;
            seed = seed * 1664525u + 1013904223u;
            int k = (seed >> 8) % K;
            if (host[(size_t)row * K + k] != cpu_ref(a, b, C, K, row, k)) {
                printf("[FAIL] %s row=%d k=%d got %d want %d\n", cs.tag, row, k,
                       host[(size_t)row * K + k], cpu_ref(a, b, C, K, row, k));
                ok = false;
            }
        }
        if (ok) printf("  spot-check 16 elems OK\n");
    }
    return 0;
}
