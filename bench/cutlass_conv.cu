#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "quant.h"
#include "reference.h"

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = int32_t;
using ElementAcc = int32_t;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAcc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    #ifndef TB_N
#define TB_N 128
#endif
#ifndef WB_N
#define WB_N 64
#endif
    cutlass::gemm::GemmShape<128, TB_N, 64>,
    cutlass::gemm::GemmShape<64, WB_N, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAcc, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kAnalytic>::Kernel;

using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;
using LayoutT = cutlass::layout::TensorNHWC;

static std::vector<int32_t> cpu_raw_conv(const std::vector<int8_t>& a, int cip, int cor, int cop,
                                         const std::vector<int8_t>& b) {
    std::vector<int32_t> out((size_t)HW * cor);
#pragma omp parallel for
    for (int p = 0; p < H; ++p)
        for (int q = 0; q < W; ++q) {
            const int8_t* rowp[3] = { nullptr, nullptr, nullptr };
            for (int r = 0; r < 3; ++r) {
                int h = p + r - 1;
                rowp[r] = (h >= 0 && h < H) ? a.data() + (size_t)h * W * cip : nullptr;
            }
            for (int k = 0; k < cor; ++k) {
                int acc = 0;
                for (int r = 0; r < 3; ++r) {
                    if (!rowp[r]) continue;
                    for (int s = 0; s < 3; ++s) {
                        int w = q + s - 1;
                        if (w < 0 || w >= W) continue;
                        const int8_t* ar = rowp[r] + (size_t)w * cip;
                        const int8_t* br = b.data() + (size_t)(s + 3 * r) * cip * cop + k;
                        for (int c = 0; c < cip; ++c)
                            acc += (int)ar[c] * (int)br[(size_t)c * cop];
                    }
                }
                out[((size_t)p * W + q) * cor + k] = acc;
            }
        }
    return out;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    ModelData m;
    if (!load_model("../cuda/weights.bin", "../cuda/input.bin", m)) {
        printf("model not found (run from bench/ dir)\n");
        return 1;
    }
    std::vector<int8_t> inter[4];
    std::vector<float> ref_out;
    cpu_int8(m, inter, ref_out);
    std::vector<int8_t> a0 = quant_input_nhwc16(m);

    const std::vector<int8_t>* inputs[5] = { &a0, &inter[0], &inter[1], &inter[2], &inter[3] };
    int8_t* dA; int8_t* dB; int32_t* dC;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)HW * 32));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)32 * 9 * 32));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)HW * 32 * 4));

    Conv2dFprop op;
    float total = 0;
    for (int l = 0; l < 5; ++l) {
        int cip = C_IN_PAD[l], cor = C_OUT_REAL[l], cop = C_OUT_PAD[l];
        std::vector<int8_t> w((size_t)cor * 9 * cip, 0);
        for (int k = 0; k < cor; ++k)
            for (int rs = 0; rs < 9; ++rs)
                for (int c = 0; c < cip; ++c)
                    w[((size_t)k * 9 + rs) * cip + c] =
                        m.layer[l].b[(size_t)(c + cip * rs) * cop + k];

        CUDA_CHECK(cudaMemcpy(dA, inputs[l]->data(), (size_t)HW * cip, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, w.data(), w.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dC, 0, (size_t)HW * cor * 4));

        cutlass::conv::Conv2dProblemSize ps(
            1, H, W, cip,
            cor, 3, 3,
            H, W,
            1, 1,
            1, 1,
            1, 1,
            cutlass::conv::Mode::kCrossCorrelation);
        typename Conv2dFprop::Arguments args{
            ps,
            {(int8_t*)dA, LayoutT(LayoutT::packed(ps.activation_extent()))},
            {(int8_t*)dB, LayoutT(LayoutT::packed(ps.filter_extent()))},
            {(int32_t*)dC, LayoutT(LayoutT::packed(ps.output_extent()))},
            {(int32_t*)dC, LayoutT(LayoutT::packed(ps.output_extent()))},
            {1.0f, 0.0f}};
        if (op.can_implement(args) != cutlass::Status::kSuccess) {
            printf("L%d cannot implement\n", l);
            return 1;
        }
        if (op.initialize(args, nullptr) != cutlass::Status::kSuccess) {
            printf("L%d init fail\n", l);
            return 1;
        }
        for (int i = 0; i < 3; ++i)
            if (op.run(nullptr) != cutlass::Status::kSuccess) { printf("warmup fail\n"); return 1; }
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        const int iters = 20;
        float ms = 0;
        cudaEventRecord(e0);
        for (int i = 0; i < iters; ++i)
            if (op.run(nullptr) != cutlass::Status::kSuccess) { printf("run fail\n"); return 1; }
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1);
        ms /= iters;
        total += ms;
        cudaEventDestroy(e0); cudaEventDestroy(e1);

        std::vector<int32_t> host((size_t)HW * cor);
        CUDA_CHECK(cudaMemcpy(host.data(), dC, host.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<int32_t> ref = cpu_raw_conv(*inputs[l], cip, cor, cop, m.layer[l].b);
        size_t bad = (size_t)-1;
        for (size_t i = 0; i < host.size(); ++i)
            if (host[i] != ref[i]) { bad = i; break; }
        if (bad != (size_t)-1) {
            printf("[FAIL] L%d raw s32 mismatch at %zu: got %d want %d\n",
                   l, bad, host[bad], ref[bad]);
            return 1;
        }
        double bytes = (double)HW * (cip + cor);
        printf("cutlass L%d: %7.3f ms  %6.1f GB/s(in+out)  raw-s32 bit-exact OK\n",
               l, ms, bytes / (ms * 1e6));
    }
    printf("cutlass total: %.3f ms\n", total);
    return 0;
}



