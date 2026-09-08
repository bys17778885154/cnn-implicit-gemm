#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

__global__ void frag_test_kernel(const int8_t* A, const int8_t* Bn, int32_t* D, int bmode) {
    __shared__ int8_t As[16][16];
    __shared__ int8_t Bs[8][16];
    int lane = threadIdx.x;
    for (int i = lane; i < 256; i += 32) As[i / 16][i % 16] = A[i];
    for (int i = lane; i < 128; i += 32) Bs[i / 16][i % 16] = Bn[i];
    __syncthreads();

    uint32_t a0, a1, b0;
    {
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(&As[lane % 16][0]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                     : "=r"(a0), "=r"(a1) : "r"(addr));
    }
    if (bmode == 0) {
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(&Bs[lane % 8][0]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                     : "=r"(b0) : "r"(addr));
    } else if (bmode == 1) {
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(&Bs[lane % 8][0]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                     : "=r"(b0) : "r"(addr));
    } else {
        b0 = *(uint32_t*)&Bs[lane / 4][(lane % 4) * 4];
    }

    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                 : "r"(a0), "r"(a1), "r"(b0));
    int r1 = lane / 4, c1 = (lane % 4) * 2;
    D[r1 * 8 + c1] = d0;
    D[r1 * 8 + c1 + 1] = d1;
    D[(r1 + 8) * 8 + c1] = d2;
    D[(r1 + 8) * 8 + c1 + 1] = d3;
}

int run_frag_test() {
    const int N = 16 * 16;
    std::vector<int8_t> hA(N), hB(8 * 16);
    for (int i = 0; i < N; ++i) hA[i] = (int8_t)((i * 7 + 3) % 251 - 125);
    for (int i = 0; i < 128; ++i) hB[i] = (int8_t)((i * 11 + 5) % 241 - 120);
    std::vector<int32_t> ref(16 * 8);
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n) {
            int acc = 0;
            for (int k = 0; k < 16; ++k)
                acc += (int)hA[m * 16 + k] * (int)hB[n * 16 + k];
            ref[m * 8 + n] = acc;
        }
    int8_t* dA; int8_t* dB; int32_t* dD;
    cudaMalloc(&dA, N); cudaMalloc(&dB, 128); cudaMalloc(&dD, 16 * 8 * 4);
    cudaMemcpy(dA, hA.data(), N, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), 128, cudaMemcpyHostToDevice);
    const char* names[3] = { "ldmatrix.x1.trans", "ldmatrix.x1", "plain ld.shared" };
    int winner = -1;
    for (int mode = 0; mode < 3; ++mode) {
        cudaMemset(dD, 0, 16 * 8 * 4);
        frag_test_kernel<<<1, 32>>>(dA, dB, dD, mode);
        std::vector<int32_t> got(16 * 8);
        cudaMemcpy(got.data(), dD, 16 * 8 * 4, cudaMemcpyDeviceToHost);
        cudaError_t err = cudaGetLastError();
        bool ok = (err == cudaSuccess);
        int firstbad = -1;
        for (int i = 0; i < 128 && ok; ++i)
            if (got[i] != ref[i]) { ok = false; firstbad = i; }
        printf("frag test B-mode %d (%s): %s%s\n", mode, names[mode], ok ? "PASS" : "FAIL",
               ok ? "" : (err != cudaSuccess ? cudaGetErrorString(err) :
                          (firstbad >= 0 ? " (value mismatch)" : "")));
        if (ok && winner < 0) winner = mode;
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    return winner < 0 ? 1 : 0;
}
