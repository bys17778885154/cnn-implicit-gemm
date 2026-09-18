#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include "conv_kernel.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static const int H = 360, W = 796, HW = H * W;

template <int TB_M, int TB_N, int WB_M, int WB_N, int STAGES, int ITER_OPT>
struct CfgT {
    static constexpr cutlass::conv::IteratorAlgorithm kIter =
        ITER_OPT ? cutlass::conv::IteratorAlgorithm::kOptimized
                 : cutlass::conv::IteratorAlgorithm::kAnalytic;
    using Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        int8_t, cutlass::layout::TensorNHWC,
        int8_t, cutlass::layout::TensorNHWC,
        int32_t, cutlass::layout::TensorNHWC,
        int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<TB_M, TB_N, 64>,
        cutlass::gemm::GemmShape<WB_M, WB_N, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        STAGES, cutlass::arch::OpMultiplyAddSaturate, kIter>::Kernel;
    using Op = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
};

template <int CIN, int COUT, int CHUNK>
static double run_ours(const int8_t* dA, const std::vector<int8_t*>& dBch, int32_t* dC) {
    dim3 g((HW + 127) / 128), t(128);
    auto launch_all = [&]() {
        for (int ch = 0; ch < COUT / CHUNK; ++ch)
            conv_mma32_kernel<CIN, CHUNK, COUT, false, true>
                <<<g, t>>>(dA, dC, dBch[ch], nullptr, nullptr, dC, ch * CHUNK);
    };
    launch_all();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    double best = 1e9;
    for (int r = 0; r < 3; ++r) {
        cudaEventRecord(e0);
        for (int i = 0; i < 20; ++i) launch_all();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        best = std::min(best, ms / 20.0);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return best;
}
template <typename Op>
static double run_cutlass(int C, int K, const int8_t* dA, const int8_t* dB, int32_t* dC) {
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
    if (op.can_implement(args) != cutlass::Status::kSuccess) return -1.0;
    if (op.initialize(args, nullptr) != cutlass::Status::kSuccess) return -1.0;
    for (int i = 0; i < 3; ++i) op.run(nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    double best = 1e9;
    for (int r = 0; r < 3; ++r) {
        cudaEventRecord(e0);
        for (int i = 0; i < 20; ++i) op.run(nullptr);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        best = std::min(best, ms / 20.0);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return best;
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
            const int8_t* br = &b[(((size_t)k * 9 + r * 3 + s) * C)];
            for (int c = 0; c < C; ++c) acc += (int)ar[c] * (int)br[c];
        }
    }
    return acc;
}

static uint32_t lcg = 12345;
static int8_t rnd8() { lcg = lcg * 1664525u + 1013904223u; return (int8_t)(lcg >> 24); }

template <int C, int K>
static void verify(const std::vector<int8_t>& a, const std::vector<int8_t>& b, const int32_t* dC,
                   const char* tag) {
    std::vector<int32_t> host((size_t)HW * K);
    CUDA_CHECK(cudaMemcpy(host.data(), dC, host.size() * 4, cudaMemcpyDeviceToHost));
    uint32_t seed = 999;
    int rows[] = { 0, W - 1, W, HW - W, HW - 1, HW - W + 1, 5 * W + 3, 129 * W + 710 };
    for (int k : {0, 1, K / 2, K - 1})
        for (int row : rows)
            if (host[(size_t)row * K + k] != cpu_ref(a, b, C, K, row, k)) {
                printf("[FAIL] %s row=%d k=%d\n", tag, row, k);
                exit(1);
            }
    for (int t = 0; t < 2000; ++t) {
        seed = seed * 1664525u + 1013904223u;
        int row = (seed >> 8) % HW;
        seed = seed * 1664525u + 1013904223u;
        int k = (seed >> 8) % K;
        if (host[(size_t)row * K + k] != cpu_ref(a, b, C, K, row, k)) {
            printf("[FAIL] %s row=%d k=%d\n", tag, row, k);
            exit(1);
        }
    }
    printf("  %s: 2032-point verify (incl. corners/borders) OK\n", tag);
}

template <int C, int K>
static void shape_case(const char* name) {
    printf("=== %s (C=%d K=%d) ===\n", name, C, K);
    std::vector<int8_t> a((size_t)HW * C), b((size_t)K * 9 * C);
    for (auto& v : a) v = rnd8();
    for (auto& v : b) v = rnd8();
    std::vector<int8_t> bg((size_t)K * 9 * C);
    for (int k = 0; k < K; ++k)
        for (int rs = 0; rs < 9; ++rs)
            for (int c = 0; c < C; ++c)
                bg[(size_t)(c + C * rs) * K + k] = b[(((size_t)k * 9 + rs) * C) + c];

    int8_t* dA; int8_t* dB; int8_t* dBg; int32_t* dC;
    CUDA_CHECK(cudaMalloc(&dA, a.size()));
    CUDA_CHECK(cudaMalloc(&dB, b.size()));
    CUDA_CHECK(cudaMalloc(&dBg, bg.size()));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)HW * K * 4));
    CUDA_CHECK(cudaMemcpy(dA, a.data(), a.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, b.data(), b.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBg, bg.data(), bg.size(), cudaMemcpyHostToDevice));

    struct { const char* tag; double (*fn)(int, int, const int8_t*, const int8_t*, int32_t*); } cs[] = {
        {"cutlass 128x128 ana", +[](int c, int k, const int8_t* x, const int8_t* y, int32_t* z) { return run_cutlass<CfgT<128,128,64,64,3,0>::Op>(c,k,x,y,z); }},
        {"cutlass 128x128 opt", +[](int c, int k, const int8_t* x, const int8_t* y, int32_t* z) { return run_cutlass<CfgT<128,128,64,64,3,1>::Op>(c,k,x,y,z); }},
        {"cutlass 128x64  ana", +[](int c, int k, const int8_t* x, const int8_t* y, int32_t* z) { return run_cutlass<CfgT<128,64,64,32,3,0>::Op>(c,k,x,y,z); }},
        {"cutlass 128x64  opt", +[](int c, int k, const int8_t* x, const int8_t* y, int32_t* z) { return run_cutlass<CfgT<128,64,64,32,3,1>::Op>(c,k,x,y,z); }},
        {"cutlass 64x128  ana", +[](int c, int k, const int8_t* x, const int8_t* y, int32_t* z) { return run_cutlass<CfgT<64,128,32,64,3,0>::Op>(c,k,x,y,z); }},
        {"cutlass 64x128  opt", +[](int c, int k, const int8_t* x, const int8_t* y, int32_t* z) { return run_cutlass<CfgT<64,128,32,64,3,1>::Op>(c,k,x,y,z); }},
    };
    double best_c = 1e9; const char* best_tag = "";
    for (auto& cfg : cs) {
        CUDA_CHECK(cudaMemset(dC, 0, (size_t)HW * K * 4));
        double ms = cfg.fn(C, K, dA, dB, dC);
        if (ms < 0) { printf("  %-20s : not implementable\n", cfg.tag); continue; }
        verify<C, K>(a, b, dC, cfg.tag);
        printf("  %-20s : %.3f ms\n", cfg.tag, ms);
        if (ms < best_c) { best_c = ms; best_tag = cfg.tag; }
    }

    CUDA_CHECK(cudaMemset(dC, 0, (size_t)HW * K * 4));
    constexpr int CH = (K >= 256 || K <= 16) ? 16 : 32;
    {
        std::vector<std::vector<int8_t>> bchh(K / CH, std::vector<int8_t>((size_t)C * 9 * CH));
        for (int ch = 0; ch < K / CH; ++ch)
            for (int gk = 0; gk < C * 9; ++gk)
                for (int k = 0; k < CH; ++k)
                    bchh[ch][(size_t)gk * CH + k] = bg[(size_t)gk * K + ch * CH + k];
        std::vector<int8_t*> dBch(K / CH);
        for (int ch = 0; ch < K / CH; ++ch) {
            CUDA_CHECK(cudaMalloc(&dBch[ch], bchh[ch].size()));
            CUDA_CHECK(cudaMemcpy(dBch[ch], bchh[ch].data(), bchh[ch].size(), cudaMemcpyHostToDevice));
        }
        double ms_nt = run_ours<C, K, CH>(dA, dBch, dC);
        verify<C, K>(a, b, dC, "ours-ntiled");
        printf("  %-20s : %.3f ms  (chunk %d)\n", "ours ntiled", ms_nt, CH);
        for (int ch = 0; ch < K / CH; ++ch) cudaFree(dBch[ch]);

        double gops = 2.0 * HW * K * 9.0 * C;
        printf("  >>> best cutlass: %s %.3f ms (%.0f GOP/s, util %.0f%%) | ours-ntiled: %.3f ms (%.0f GOP/s, util %.0f%%)\n",
               best_tag, best_c, gops / best_c / 1e6, gops / best_c / 1e6 / 226000.0 * 100,
               ms_nt, gops / ms_nt / 1e6, gops / ms_nt / 1e6 / 226000.0 * 100);
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dBg); cudaFree(dC);
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    shape_case<16, 16>("small L0");
    shape_case<16, 32>("small L1");
    shape_case<32, 16>("small L2");
    shape_case<64, 64>("large 64");
    shape_case<128, 128>("large 128");
    shape_case<256, 256>("large 256");
    return 0;
}




