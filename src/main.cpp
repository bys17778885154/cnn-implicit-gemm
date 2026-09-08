#include "gpu.h"
#include "reference.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static GpuModel gm;
static ModelData mdl;
static std::vector<int8_t> ref_inter[4];
static std::vector<float> ref_out;
static std::vector<float> ref_f32;
static bool g_has_tiled = true;
static bool g_has_mma = false;

static size_t align256(size_t x) { return (x + 255) / 256 * 256; }

static void gpu_setup(const ModelData& m) {
    size_t total = 0;
    size_t seg[19];
    int n = 0;
    for (int l = 0; l < 5; ++l) {
        seg[n++] = align256(m.layer[l].b.size());
        seg[n++] = align256((size_t)m.layer[l].bias_q.size() * 4);
        seg[n++] = align256((size_t)m.layer[l].mult.size() * 4);
    }
    seg[n++] = align256((size_t)HW * 16);
    seg[n++] = align256((size_t)HW * 16);
    seg[n++] = align256((size_t)HW * 32);
    seg[n++] = align256((size_t)HW * 4 * 4);
    for (int i = 0; i < n; ++i) total += seg[i];
    cudaMalloc(&gm.dev, total);
    uint8_t* p = gm.dev;
    int idx = 0;
    for (int l = 0; l < 5; ++l) {
        gm.b[l] = (int8_t*)p;
        cudaMemcpy((void*)gm.b[l], m.layer[l].b.data(), m.layer[l].b.size(), cudaMemcpyHostToDevice);
        p += seg[idx++];
        gm.bq[l] = (int32_t*)p;
        cudaMemcpy((void*)gm.bq[l], m.layer[l].bias_q.data(), m.layer[l].bias_q.size() * 4, cudaMemcpyHostToDevice);
        p += seg[idx++];
        gm.mult[l] = (float*)p;
        cudaMemcpy((void*)gm.mult[l], m.layer[l].mult.data(), m.layer[l].mult.size() * 4, cudaMemcpyHostToDevice);
        p += seg[idx++];
    }
    gm.a0 = (int8_t*)p; p += seg[idx++];
    gm.x[0] = (int8_t*)p; p += seg[idx++];
    gm.x[1] = (int8_t*)p; p += seg[idx++];
    gm.out = (float*)p;
    cudaMemset(gm.x[0], 0, (size_t)HW * 16);
    cudaMemset(gm.x[1], 0, (size_t)HW * 32);
}

static void* layer_out_buf(int l) {
    static void* t[5] = { nullptr };
    t[0] = gm.x[0]; t[1] = gm.x[1]; t[2] = gm.x[0]; t[3] = gm.x[1]; t[4] = gm.out;
    return t[l];
}

static bool validate_impl(const char* name, ChainFn fn) {
    float ms[5];
    for (int l = 0; l < 5; ++l) {
        fn(gm, ms, l);
        if (l < 4) {
            size_t bytes = (size_t)HW * C_OUT_PAD[l];
            std::vector<int8_t> host(bytes);
            cudaMemcpy(host.data(), layer_out_buf(l), bytes, cudaMemcpyDeviceToHost);
            if (memcmp(host.data(), ref_inter[l].data(), bytes) != 0) {
                size_t bad = 0;
                while (bad < bytes && host[bad] == ref_inter[l][bad]) ++bad;
                printf("[FAIL] %s layer %d first mismatch at byte %zu (row %zu ch %d): got %d want %d\n",
                       name, l, bad, bad / C_OUT_PAD[l], (int)(bad % C_OUT_PAD[l]),
                       (int)host[bad], (int)ref_inter[l][bad]);
                return false;
            }
        } else {
            size_t bytes = (size_t)HW * 4 * 4;
            std::vector<float> host(HW * 4);
            cudaMemcpy(host.data(), gm.out, bytes, cudaMemcpyDeviceToHost);
            size_t bad = (size_t)-1;
            for (size_t i = 0; i < host.size(); ++i)
                if (host[i] != ref_out[i]) { bad = i; break; }
            if (bad != (size_t)-1) {
                printf("[FAIL] %s layer 4 mismatch at %zu: got %.7f want %.7f\n",
                       name, bad, host[bad], ref_out[bad]);
                return false;
            }
            double maxabs = 0, sum = 0, refnorm = 0;
            for (size_t i = 0; i < host.size(); ++i) {
                double d = fabs((double)host[i] - (double)ref_f32[i]);
                if (d > maxabs) maxabs = d;
                sum += d;
                refnorm += fabs((double)ref_f32[i]);
            }
            printf("[ OK ] %s: int8 chain bit-exact; vs fp32: max_abs=%.5f mean_abs=%.6f (ref mean |y|=%.4f)\n",
                   name, maxabs, sum / host.size(), refnorm / host.size());
        }
    }
    return true;
}

static void bench_impl(const char* name, ChainFn fn) {
    float ms[5];
    float acc[5] = { 0, 0, 0, 0, 0 };
    for (int i = 0; i < 3; ++i) fn(gm, ms, -1);
    const int iters = 20;
    for (int i = 0; i < iters; ++i)
        for (int l = 0; l < 5; ++l) acc[l] += ms[l];
    double bytes =
        (double)HW * (16 + 16) + (double)HW * (16 + 32) + (double)HW * (32 + 16) +
        (double)HW * (16 + 16) + (double)HW * (16 + 16);
    double total = 0;
    printf("%-7s", name);
    for (int l = 0; l < 5; ++l) {
        double v = acc[l] / iters;
        total += v;
        printf("  L%d=%7.3fms", l, v);
    }
    printf("  total=%7.3fms  %.1f GB/s\n", total, bytes / (total * 1e6));
}

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "test";
    if (!load_model("weights.bin", "input.bin", mdl)) {
        printf("generating weights.bin / input.bin ...\n");
        gen_model("weights.bin", "input.bin");
        load_model("weights.bin", "input.bin", mdl);
    }
    cpu_int8(mdl, ref_inter, ref_out);
    cpu_fp32(mdl, ref_f32);
    std::vector<int8_t> a0 = quant_input_nhwc16(mdl);
    gpu_setup(mdl);
    cudaMemcpy((void*)gm.a0, a0.data(), a0.size(), cudaMemcpyHostToDevice);

    struct Impl { const char* name; ChainFn fn; bool enabled; };
    Impl impls[3] = {
        { "naive", conv5_naive_chain, true },
        { "tiled", conv5_tiled_chain, g_has_tiled },
        { "mma",   conv5_mma_chain,   g_has_mma },
    };

    if (mode == "test" || mode == "all") {
        if (run_frag_test() != 0) { printf("frag test FAILED`n"); return 1; }
        bool ok = true;
        for (auto& im : impls)
            if (im.enabled) ok = validate_impl(im.name, im.fn) && ok;
        if (!ok) return 1;
    }
    if (mode == "bench" || mode == "all") {
        for (auto& im : impls)
            if (im.enabled) bench_impl(im.name, im.fn);
    }
    if (mode == "gen") printf("done\n");
    return 0;
}


