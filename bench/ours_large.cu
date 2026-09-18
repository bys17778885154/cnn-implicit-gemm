#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static const int H = 360, W = 796, HW = H * W;

template <int CIN, int COUT>
__global__ void conv_mma_large_kernel(const int8_t* __restrict__ a, int32_t* __restrict__ yv,
                                      const int8_t* __restrict__ B) {
    constexpr int KS = CIN * 9;
    constexpr int BST = KS + ((KS / 16) % 2 == 0 ? 16 : 0);
    constexpr int KSTEPS = KS / 32;
    constexpr int NT = COUT / 8;

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
            if (h >= 0 && w >= 0) {
                v0 = *(const int4*)(a + ((size_t)h * W + w) * CIN);
                v1 = *(const int4*)(a + ((size_t)h * W + w) * CIN + 16);
            }
        }
        *(int4*)&As[0][tid][((tid >> 2) & 1) * 16] = v0;
        *(int4*)&As[0][tid][(((tid >> 2) & 1) ^ 1) * 16] = v1;
    }
    __syncthreads();

    int warp = tid / 32, lane = tid % 32;
    int wbase = warp * 32;
    int d[NT][2][4];
#pragma unroll
    for (int ni = 0; ni < NT; ++ni)
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int j = 0; j < 4; ++j) d[ni][mi][j] = 0;

#pragma unroll 1
    for (int ks = 0; ks < KSTEPS; ++ks) {
        int4 n0 = make_int4(0, 0, 0, 0), n1 = make_int4(0, 0, 0, 0);
        if (ks + 1 < KSTEPS && row0 < HW) {
            int rsp = (ks + 1) / (CIN / 32);
            int c0 = ((ks + 1) % (CIN / 32)) * 32;
            int h = p0 + rsp / 3 - 1, w = q0 + rsp % 3 - 1;
            if (h >= 0 && h < H && w >= 0 && w < W) {
                n0 = *(const int4*)(a + ((size_t)h * W + w) * CIN + c0);
                n1 = *(const int4*)(a + ((size_t)h * W + w) * CIN + c0 + 16);
            }
        }

        int buf = ks & 1;
        uint32_t af[2][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            int rabs = wbase + mi * 16 + (lane % 8) + ((lane >> 3) & 1) * 8;
            int phys = (((lane >> 4) & 1) ^ ((rabs >> 2) & 1)) * 16;
            uint32_t addr = (uint32_t)__cvta_generic_to_shared(&As[buf][rabs][phys]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(af[mi][0]), "=r"(af[mi][1]), "=r"(af[mi][2]), "=r"(af[mi][3]) : "r"(addr));
        }
        uint32_t b[NT][2];
#pragma unroll
        for (int p = 0; p < NT / 2; ++p) {
            int nrow = (2 * p + (lane >> 4)) * 8 + (lane % 8);
            uint32_t addr = (uint32_t)__cvta_generic_to_shared(&Bs[nrow][ks * 32 + ((lane >> 3) & 1) * 16]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(b[2 * p][0]), "=r"(b[2 * p][1]), "=r"(b[2 * p + 1][0]), "=r"(b[2 * p + 1][1]) : "r"(addr));
        }
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < NT; ++ni)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+r"(d[ni][mi][0]), "+r"(d[ni][mi][1]), "+r"(d[ni][mi][2]), "+r"(d[ni][mi][3])
                             : "r"(af[mi][0]), "r"(af[mi][1]), "r"(af[mi][2]), "r"(af[mi][3]),
                               "r"(b[ni][0]), "r"(b[ni][1]));

        if (ks + 1 < KSTEPS) {
            *(int4*)&As[(ks + 1) & 1][tid][((tid >> 2) & 1) * 16] = n0;
            *(int4*)&As[(ks + 1) & 1][tid][(((tid >> 2) & 1) ^ 1) * 16] = n1;
        }
        __syncthreads();
    }

    int mbase = blockIdx.x * 128 + warp * 32;
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            int r1 = mbase + mi * 16 + lane / 4;
            int c1 = ni * 8 + (lane % 4) * 2;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                int rr = r1 + (e / 2) * 8;
                int k = c1 + e % 2;
                if (rr < HW) yv[(size_t)rr * COUT + k] = d[ni][mi][e];
            }
        }
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
    for (int i = 0; i < 3; ++i) conv_mma_large_kernel<C, K><<<g, t>>>(dA, dC, dB);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < 20; ++i) conv_mma_large_kernel<C, K><<<g, t>>>(dA, dC, dB);
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


