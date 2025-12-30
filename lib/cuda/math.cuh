#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace ac4k {

// __device__ __forceinline__ float2 half2_to_floatx2(half2 in) {
//   float2 out;
//   asm("cvt.f32.f16 %0, %1;" : "=f"(out.x) : "h"(in.x));
//   asm("cvt.f32.f16 %0, %1;" : "=f"(out.y) : "h"(in.y));
//   return out;
// }

} // namespace ac4k
