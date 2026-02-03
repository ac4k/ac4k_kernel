// Copyright 2024-2026 AC4K Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>

#include "ac4k_kernel/ops.h"

//===----------------------------------------------------------------------===//
// Compile-time Architecture Selection
//
// The architecture is determined at build time via -DAC4K_ARCH_SMXXX=1
// This enables zero-overhead dispatch - no runtime branching.
//===----------------------------------------------------------------------===//

#if defined(AC4K_ARCH_SM120)
constexpr const char *kArchName = "sm120";
constexpr const char *kArchDescription = "NVIDIA Blackwell (RTX 5090)";
#elif defined(AC4K_ARCH_SM100)
constexpr const char *kArchName = "sm100";
constexpr const char *kArchDescription = "NVIDIA Blackwell (B200/B100)";
#elif defined(AC4K_ARCH_SM90A)
constexpr const char *kArchName = "sm90a";
constexpr const char *kArchDescription = "NVIDIA Hopper (H100/H200)";
#elif defined(AC4K_ARCH_SM89)
constexpr const char *kArchName = "sm89";
constexpr const char *kArchDescription = "NVIDIA Ada Lovelace (RTX 4090)";
#else
// Default to sm120 for backward compatibility
constexpr const char *kArchName = "sm120";
constexpr const char *kArchDescription = "NVIDIA Blackwell (RTX 5090) [default]";
#endif

//===----------------------------------------------------------------------===//
// Python Module Registration
//===----------------------------------------------------------------------===//

PYBIND11_MODULE(_cuda_ops, m) {
  m.doc() = "AC4K CUDA Operators - Zero Overhead Architecture-Specific Kernels";

  // Module metadata (for introspection)
  m.attr("__arch__") = kArchName;
  m.attr("__arch_description__") = kArchDescription;
  m.attr("__backend__") = "cuda";
  m.attr("__version__") = "0.1.0";

  //-------------------------------------------------------------------------
  // Attention Operators
  //-------------------------------------------------------------------------
  m.def("mha_nvfp4_fwd", &ac4k::mha_nvfp4_fwd,
        "Multi-Head Attention Forward (NVFP4)\n\n"
        "Args:\n"
        "    o: Output tensor [B, H, N, D], bfloat16\n"
        "    q, q_sf, q_global_scale: Quantized query\n"
        "    k, k_sf, k_global_scale: Quantized key\n"
        "    v, v_sf, v_global_scale: Quantized value\n"
        "    sm_scale: Softmax scale factor");

  m.def("mha_int8_x_fp8_fwd", &ac4k::mha_int8_x_fp8_fwd,
        "Multi-Head Attention Forward (QK=INT8, PV=FP8)");

  //-------------------------------------------------------------------------
  // Quantization Operators
  //-------------------------------------------------------------------------
  m.def("quantize_nvfp4", &ac4k::quantize_nvfp4,
        "Quantize bfloat16 tensor to NVFP4 format\n\n"
        "Args:\n"
        "    out: Output tensor (nvfp4 packed)\n"
        "    sf: Scale factors\n"
        "    input: Input tensor (bfloat16)\n"
        "    global_scale: Global scale factor\n"
        "    cross_dim, reduce_dim: Dimension indices\n"
        "    swizzle: Enable memory swizzle");

  m.def("quantize_fp8", &ac4k::quantize_fp8,
        "Quantize bfloat16 tensor to FP8 format");

  m.def("quantize_int8", &ac4k::quantize_int8,
        "Quantize bfloat16 tensor to INT8 format");

  //-------------------------------------------------------------------------
  // Linear Operators
  //-------------------------------------------------------------------------
  m.def("linear_nvfp4", &ac4k::linear_nvfp4,
        "NVFP4 Linear (fully-connected layer)\n\n"
        "y = x @ weight.T + bias");

  //-------------------------------------------------------------------------
  // RoPE Operators
  //-------------------------------------------------------------------------
  m.def("rope3d", &ac4k::rope3d,
        "3D Rotary Position Embedding\n\n"
        "Args:\n"
        "    x: Input tensor [B, S, N, D], bfloat16\n"
        "    grid_sizes: Grid sizes [B, 3], int32\n"
        "    freqs: Frequency table [max_pos, C], complex128\n"
        "    output: Output tensor [B, S, N, D], bfloat16");
}
