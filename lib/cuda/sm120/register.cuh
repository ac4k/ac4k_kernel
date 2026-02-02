#pragma once

#include <cuda.h>

//===----------------------------------------------------------------------===//
// Heterogeneous allocation of registers for producer & consumer.
//===----------------------------------------------------------------------===//

namespace ac4k::sm120 {

template <uint32_t RegCount> __forceinline__ __device__ void reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount> __forceinline__ __device__ void reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

} // namespace ac4k::sm120
