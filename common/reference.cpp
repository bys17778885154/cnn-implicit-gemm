#include "reference.h"
#ifdef _OPENMP
#include <omp.h>
#endif

std::vector<int8_t> quant_input_nhwc16(const ModelData& m) {
    std::vector<int8_t> a((size_t)HW * 16, 0);
    float s = m.a_scale[0];
#pragma omp parallel for
    for (int p = 0; p < H; ++p)
        for (int q = 0; q < W; ++q)
            for (int c = 0; c < 4; ++c) {
                float v = m.input[(size_t)c * HW + (size_t)p * W + q] / s;
                a[((size_t)p * W + q) * 16 + c] = (int8_t)clamp_i32(quant_rn(v), -128, 127);
            }
    return a;
}

static void conv_layer_cpu(const int8_t* a, int cip, int8_t* y, int cor, int cop,
                           const int8_t* B, const int32_t* bq, const float* mult) {
#pragma omp parallel for
    for (int p = 0; p < H; ++p)
        for (int q = 0; q < W; ++q) {
            const int8_t* rowptr[3] = { nullptr, nullptr, nullptr };
            for (int r = 0; r < 3; ++r) {
                int h = p + r - 1;
                rowptr[r] = (h >= 0 && h < H) ? a + (size_t)h * W * cip : nullptr;
            }
            for (int k = 0; k < cor; ++k) {
                int acc = 0;
                for (int r = 0; r < 3; ++r) {
                    if (!rowptr[r]) continue;
                    for (int s = 0; s < 3; ++s) {
                        int w = q + s - 1;
                        if (w < 0 || w >= W) continue;
                        const int8_t* ar = rowptr[r] + (size_t)w * cip;
                        const int8_t* br = B + (size_t)(s + 3 * r) * cip * cop + k;
                        for (int c = 0; c < cip; ++c)
                            acc += (int)ar[c] * (int)br[(size_t)c * cop];
                    }
                }
                acc += bq[k];
                int v = quant_rn((float)acc * mult[k]);
                y[((size_t)p * W + q) * cop + k] = (int8_t)clamp_i32(v, 0, 127);
            }
        }
}

void cpu_int8(const ModelData& m, std::vector<int8_t> inter[4], std::vector<float>& out) {
    std::vector<int8_t> a = quant_input_nhwc16(m);
    for (int l = 0; l < NUM_LAYERS; ++l) {
        const LayerData& L = m.layer[l];
        if (l + 1 < NUM_LAYERS) {
            inter[l].assign((size_t)HW * C_OUT_PAD[l], 0);
            conv_layer_cpu(a.data(), C_IN_PAD[l], inter[l].data(), C_OUT_REAL[l], C_OUT_PAD[l],
                           L.b.data(), L.bias_q.data(), L.mult.data());
            a = inter[l];
        } else {
            out.assign((size_t)HW * 4, 0.0f);
#pragma omp parallel for
            for (int p = 0; p < H; ++p)
                for (int q = 0; q < W; ++q)
                    for (int k = 0; k < 4; ++k) {
                        int acc = 0;
                        for (int r = 0; r < 3; ++r) {
                            int h = p + r - 1;
                            if (h < 0 || h >= H) continue;
                            for (int s = 0; s < 3; ++s) {
                                int w = q + s - 1;
                                if (w < 0 || w >= W) continue;
                                const int8_t* ar = a.data() + ((size_t)h * W + w) * C_IN_PAD[l];
                                const int8_t* br = L.b.data() + (size_t)(s + 3 * r) * C_IN_PAD[l] * C_OUT_PAD[l] + k;
                                for (int c = 0; c < C_IN_PAD[l]; ++c)
                                    acc += (int)ar[c] * (int)br[(size_t)c * C_OUT_PAD[l]];
                            }
                        }
                        acc += L.bias_q[k];
                        out[((size_t)p * W + q) * 4 + k] = (float)acc * L.mult[k];
                    }
        }
    }
}

void cpu_fp32(const ModelData& m, std::vector<float>& out_nhwc) {
    std::vector<float> cur((size_t)HW * 4), nxt;
    for (int p = 0; p < H; ++p)
        for (int q = 0; q < W; ++q)
            for (int c = 0; c < 4; ++c)
                cur[((size_t)p * W + q) * 4 + c] = m.input[(size_t)c * HW + (size_t)p * W + q];
    for (int l = 0; l < NUM_LAYERS; ++l) {
        int ci = C_IN_REAL[l], kr = C_OUT_REAL[l];
        nxt.assign((size_t)HW * kr, 0.0f);
        const LayerData& L = m.layer[l];
#pragma omp parallel for
        for (int p = 0; p < H; ++p)
            for (int q = 0; q < W; ++q)
                for (int k = 0; k < kr; ++k) {
                    float swk = (l + 1 < NUM_LAYERS)
                        ? L.mult[k] * m.a_scale[l + 1] / m.a_scale[l]
                        : L.mult[k] / m.a_scale[l];
                    float acc = 0.0f;
                    for (int r = 0; r < 3; ++r) {
                        int h = p + r - 1;
                        if (h < 0 || h >= H) continue;
                        for (int s = 0; s < 3; ++s) {
                            int w = q + s - 1;
                            if (w < 0 || w >= W) continue;
                            const float* ar = cur.data() + ((size_t)h * W + w) * ci;
                            const int8_t* br = L.b.data() + (size_t)(s + 3 * r) * C_IN_PAD[l] * C_OUT_PAD[l] + k;
                            for (int c = 0; c < ci; ++c)
                                acc += ar[c] * ((float)br[(size_t)c * C_OUT_PAD[l]] * swk);
                        }
                    }
                    acc += (float)L.bias_q[k] * swk * m.a_scale[l];
                    if (l + 1 < NUM_LAYERS) nxt[((size_t)p * W + q) * kr + k] = acc > 0.0f ? acc : 0.0f;
                    else nxt[((size_t)p * W + q) * kr + k] = acc;
                }
        cur.swap(nxt);
    }
    out_nhwc = cur;
}
