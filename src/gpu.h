#pragma once
#include "quant.h"

struct GpuModel {
    uint8_t* dev = nullptr;
    const int8_t* b[5] = {};
    const int32_t* bq[5] = {};
    const float* mult[5] = {};
    const int8_t* a0 = nullptr;
    int8_t* x[2] = {};
    float* out = nullptr;
};

using ChainFn = void(*)(const GpuModel& g, float* layer_ms, int last_layer);

void conv5_naive_chain(const GpuModel& g, float* layer_ms, int last_layer);
void conv5_tiled_chain(const GpuModel& g, float* layer_ms, int last_layer);
void conv5_mma_chain(const GpuModel& g, float* layer_ms, int last_layer);
void conv5_mma32_chain(const GpuModel& g, float* layer_ms, int last_layer);
int run_frag_test();
