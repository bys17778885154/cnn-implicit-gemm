#include "../src/quant.h"
#include <cstdio>
#include <random>

static float conv_one(const std::vector<int8_t>& aq, int cip, const std::vector<int8_t>& wq,
                      int k, int ci, int p, int q, int bqk) {
    int acc = 0;
    for (int rs = 0; rs < 9; ++rs) {
        int r = rs / 3, s = rs % 3;
        int h = p + r - 1, w = q + s - 1;
        if (h < 0 || h >= H || w < 0 || w >= W) continue;
        const int8_t* ar = aq.data() + ((size_t)h * W + w) * cip;
        for (int c = 0; c < ci; ++c)
            acc += (int)ar[c] * (int)wq[(size_t)k * ci * 9 + (size_t)c * 9 + rs];
    }
    return (float)(acc + bqk);
}

void gen_model(const char* wpath, const char* ipath) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    ModelData m;
    std::vector<float> wf[NUM_LAYERS];
    std::vector<float> biasf[NUM_LAYERS];
    std::vector<int8_t> wq[NUM_LAYERS];
    std::vector<float> s_w[NUM_LAYERS];

    m.input.resize((size_t)4 * HW);
    for (auto& v : m.input) v = 1.5f * U(rng);
    float imax = 0.0f;
    for (auto v : m.input) { float av = fabsf(v); if (av > imax) imax = av; }
    m.a_scale[0] = imax / 127.0f;

    for (int l = 0; l < NUM_LAYERS; ++l) {
        int kr = C_OUT_REAL[l], ci = C_IN_REAL[l];
        wf[l].resize((size_t)kr * ci * 9);
        float gain = sqrtf(6.0f / (float)(9 * ci));
        for (auto& v : wf[l]) v = U(rng) * gain;
        biasf[l].resize(kr);
        for (auto& v : biasf[l]) v = 0.5f * U(rng) * sqrtf(2.0f / (float)(9 * ci));
        wq[l].resize((size_t)kr * ci * 9);
        s_w[l].resize(kr);
        for (int k = 0; k < kr; ++k) {
            float mx = 0.0f;
            for (int i = 0; i < ci * 9; ++i) {
                float av = fabsf(wf[l][(size_t)k * ci * 9 + i]);
                if (av > mx) mx = av;
            }
            s_w[l][k] = mx / 127.0f;
            for (int i = 0; i < ci * 9; ++i)
                wq[l][(size_t)k * ci * 9 + i] =
                    (int8_t)clamp_i32(quant_rn(wf[l][(size_t)k * ci * 9 + i] / s_w[l][k]), -127, 127);
        }
    }

    std::vector<int8_t> aq((size_t)HW * 16, 0);
    for (int p = 0; p < H; ++p)
        for (int q = 0; q < W; ++q)
            for (int c = 0; c < 4; ++c)
                aq[((size_t)p * W + q) * 16 + c] =
                    (int8_t)clamp_i32(quant_rn(m.input[(size_t)c * HW + (size_t)p * W + q] / m.a_scale[0]), -128, 127);

    for (int l = 0; l < NUM_LAYERS; ++l) {
        int kr = C_OUT_REAL[l], ci = C_IN_REAL[l];
        m.layer[l].b.assign((size_t)K_GEMM[l] * C_OUT_PAD[l], 0);
        m.layer[l].bias_q.assign(C_OUT_PAD[l], 0);
        m.layer[l].mult.assign(C_OUT_PAD[l], 0.0f);
        std::vector<int32_t> bql(kr);
        for (int k = 0; k < kr; ++k)
            bql[k] = quant_rn(biasf[l][k] / (m.a_scale[l] * s_w[l][k]));

        std::vector<float> accv((size_t)HW * kr);
        float amax = 0.0f;
        for (int p = 0; p < H; ++p)
            for (int q = 0; q < W; ++q)
                for (int k = 0; k < kr; ++k) {
                    float acc = conv_one(aq, C_IN_PAD[l], wq[l], k, ci, p, q, bql[k]);
                    accv[((size_t)p * W + q) * kr + k] = acc;
                    float pos = acc * s_w[l][k];
                    if (pos > amax) amax = pos;
                }

        float s_next = (l + 1 < NUM_LAYERS)
            ? fmaxf(m.a_scale[l] * amax / 127.0f, 1e-12f)
            : 1.0f;
        for (int k = 0; k < kr; ++k) {
            m.layer[l].bias_q[k] = bql[k];
            m.layer[l].mult[k] = (l + 1 < NUM_LAYERS)
                ? s_w[l][k] * m.a_scale[l] / s_next
                : s_w[l][k] * m.a_scale[l];
            for (int gk = 0; gk < K_GEMM[l]; ++gk) {
                int c = gk % C_IN_PAD[l];
                int rs = gk / C_IN_PAD[l];
                int8_t qv = 0;
                if (c < ci)
                    qv = wq[l][(size_t)k * ci * 9 + (size_t)c * 9 + rs];
                m.layer[l].b[(size_t)gk * C_OUT_PAD[l] + k] = qv;
            }
        }

        if (l + 1 < NUM_LAYERS) {
            m.a_scale[l + 1] = s_next;
            int cop = C_OUT_PAD[l];
            std::vector<int8_t> nxt((size_t)HW * cop, 0);
            for (int p = 0; p < H; ++p)
                for (int q = 0; q < W; ++q)
                    for (int k = 0; k < kr; ++k) {
                        int v = quant_rn(accv[((size_t)p * W + q) * kr + k] * m.layer[l].mult[k]);
                        nxt[((size_t)p * W + q) * cop + k] = (int8_t)clamp_i32(v, 0, 127);
                    }
            aq.swap(nxt);
        }
    }

    FILE* fw = fopen(wpath, "wb");
    uint32_t magic = 0xC0FFEE05u;
    fwrite(&magic, 4, 1, fw);
    fwrite(m.a_scale, sizeof(float), NUM_LAYERS, fw);
    for (int l = 0; l < NUM_LAYERS; ++l) {
        fwrite(m.layer[l].b.data(), 1, m.layer[l].b.size(), fw);
        fwrite(m.layer[l].bias_q.data(), 4, m.layer[l].bias_q.size(), fw);
        fwrite(m.layer[l].mult.data(), 4, m.layer[l].mult.size(), fw);
    }
    fclose(fw);
    FILE* fi = fopen(ipath, "wb");
    fwrite(m.input.data(), 4, m.input.size(), fi);
    fclose(fi);
}

bool load_model(const char* wpath, const char* ipath, ModelData& m) {
    FILE* f = fopen(wpath, "rb");
    if (!f) return false;
    uint32_t magic = 0;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0xC0FFEE05u) { fclose(f); return false; }
    if (fread(m.a_scale, sizeof(float), NUM_LAYERS, f) != NUM_LAYERS) { fclose(f); return false; }
    for (int l = 0; l < NUM_LAYERS; ++l) {
        auto& L = m.layer[l];
        L.b.resize((size_t)K_GEMM[l] * C_OUT_PAD[l]);
        L.bias_q.resize(C_OUT_PAD[l]);
        L.mult.resize(C_OUT_PAD[l]);
        size_t rb = fread(L.b.data(), 1, L.b.size(), f);
        size_t r1 = fread(L.bias_q.data(), 4, L.bias_q.size(), f);
        size_t r2 = fread(L.mult.data(), 4, L.mult.size(), f);
        if (rb != L.b.size() || r1 != L.bias_q.size() || r2 != L.mult.size()) { fclose(f); return false; }
    }
    fclose(f);
    FILE* fi = fopen(ipath, "rb");
    if (!fi) return false;
    m.input.resize((size_t)4 * HW);
    if (fread(m.input.data(), 4, m.input.size(), fi) != m.input.size()) { fclose(fi); return false; }
    fclose(fi);
    return true;
}
