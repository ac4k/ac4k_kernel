#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace ac4k {

constexpr float LOG2E_F = 1.4426950408889634f;

//===----------------------------------------------------------------------===//
// y = e^x => y = 2^(x * log2(e))
//===----------------------------------------------------------------------===//

__forceinline__ __device__ float fast_exp(float x, float scaled) {
  scaled = scaled * LOG2E_F;
  return exp2f(x * LOG2E_F - scaled);
}

__forceinline__ __device__ float fast_exp(float x) {
  return exp2f(x * LOG2E_F);
}

//===----------------------------------------------------------------------===//
// Convert 4xFP32 to 4xFP8 (represented as one uint32_t)
//===----------------------------------------------------------------------===//

__device__ __forceinline__ uint32_t fp32x4_to_e4m3x4(float in0, float in1,
                                                     float in2, float in3) {
  uint32_t val;
  asm volatile("{\n"
               ".reg .b16 lo;\n"
               ".reg .b16 hi;\n"
               "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n"
               "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n"
               "mov.b32 %0, {lo, hi};\n"
               "}"
               : "=r"(val)
               : "f"(in0), "f"(in1), "f"(in2), "f"(in3));
  return val;
}

__device__ __forceinline__ uint32_t fp32x4_to_e4m3x4(float4 in) {
  return fp32x4_to_e4m3x4(in.x, in.y, in.z, in.w);
}

} // namespace ac4k
