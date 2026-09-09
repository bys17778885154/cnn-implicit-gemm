#pragma once
#include "quant.h"

void cpu_fp32(const ModelData& m, std::vector<float>& out_nhwc);
void cpu_int8(const ModelData& m, std::vector<int8_t> inter[4], std::vector<float>& out);
std::vector<int8_t> quant_input_nhwc16(const ModelData& m);
