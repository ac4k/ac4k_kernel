// Copyright 2024-2026 AC4K Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ac4k {

//===----------------------------------------------------------------------===//
// Backend Types
//===----------------------------------------------------------------------===//

enum class Backend { CUDA, ROCm, Unknown };

//===----------------------------------------------------------------------===//
// CUDA Architectures
//===----------------------------------------------------------------------===//

enum class CUDAArch {
  SM120, // Blackwell (RTX 5090)
  SM100, // Blackwell (B200/B100)
  SM90a, // Hopper (H100/H200)
  SM89,  // Ada Lovelace (RTX 4090)
  Unknown
};

//===----------------------------------------------------------------------===//
// ROCm Architectures
//===----------------------------------------------------------------------===//

enum class ROCmArch {
  GFX942, // MI300X
  GFX90a, // MI250X
  Unknown
};

//===----------------------------------------------------------------------===//
// Precision Types
//===----------------------------------------------------------------------===//

enum class Precision {
  FP32,
  FP16,
  BF16,
  FP8_E4M3,
  FP8_E5M2,
  INT8,
  NVFP4, // E2M1 with block scaling (Blackwell only)
  Unknown
};

//===----------------------------------------------------------------------===//
// Type Aliases
//===----------------------------------------------------------------------===//

using index_t = int64_t;

} // namespace ac4k
