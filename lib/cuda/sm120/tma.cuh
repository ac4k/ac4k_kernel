#pragma once

#include <cuda.h>

namespace ac4k::sm120 {

//===----------------------------------------------------------------------===//
// Swizzle
//===----------------------------------------------------------------------===//

template <int INNER_DIM_SIZE, int /* Bytes-Per-ELEMENT */ BPE>
__forceinline__ constexpr CUtensorMapSwizzle get_swizzle() {
  if constexpr (INNER_DIM_SIZE * BPE == 128) {
    return CU_TENSOR_MAP_SWIZZLE_128B;
  } else if constexpr (INNER_DIM_SIZE * BPE == 64) {
    return CU_TENSOR_MAP_SWIZZLE_64B;
  } else {
    return CU_TENSOR_MAP_SWIZZLE_NONE;
  }
}

} // namespace ac4k::sm120
