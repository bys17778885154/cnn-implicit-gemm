#pragma once
#include <cstdint>
#include <cmath>
#include <vector>

inline constexpr int H = 360, W = 796, HW = H * W;
inline constexpr int NUM_LAYERS = 5;
inline constexpr int C_IN_REAL[NUM_LAYERS]  = { 4, 16, 32, 16, 16 };
inline constexpr int C_IN_PAD[NUM_LAYERS]   = { 16, 16, 32, 16, 16 };
inline constexpr int C_OUT_REAL[NUM_LAYERS] = { 16, 32, 16, 16, 4 };
inline constexpr int C_OUT_PAD[NUM_LAYERS]  = { 16, 32, 16, 16, 8 };
inline constexpr int K_GEMM[NUM_LAYERS]     = { 144, 144, 288, 144, 144 };

struct LayerData {
    std::vector<int8_t> b;
    std::vector<int32_t> bias_q;
    std::vector<float> mult;
};

struct ModelData {
    float a_scale[NUM_LAYERS];
    LayerData layer[NUM_LAYERS];
    std::vector<float> input;
};

inline int quant_rn(float x) { return (int)lrintf(x); }
inline int clamp_i32(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

void gen_model(const char* wpath, const char* ipath);
bool load_model(const char* wpath, const char* ipath, ModelData& m);
