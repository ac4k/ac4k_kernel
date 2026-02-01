// Copyright 2024-2026 AC4K Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>

#include "ac4k_kernel/ops/cuda_ops.h"

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
  m.def("nvfp4_mha_fwd", &ac4k::nvfp4_mha_fwd_sm120,
        "NVFP4 Multi-Head Attention Forward\n\n"
        "Args:\n"
        "    o: Output tensor [B, H, N, D], bfloat16\n"
        "    q, q_sf, q_global_scale: Quantized query\n"
        "    k, k_sf, k_global_scale: Quantized key\n"
        "    v, v_sf, v_global_scale: Quantized value\n"
        "    sm_scale: Softmax scale factor");

  m.def("qk_int8_pv_fp8_mha_fwd", &ac4k::qk_int8_pv_fp8_mha_fwd_sm120,
        "INT8-QK FP8-PV Multi-Head Attention Forward");

  //-------------------------------------------------------------------------
  // Quantization Operators
  //-------------------------------------------------------------------------
  m.def("nvfp4_quantize", &ac4k::nvfp4_quantize_sm120,
        "Quantize bfloat16 tensor to NVFP4 format\n\n"
        "Args:\n"
        "    out: Output tensor (nvfp4 packed)\n"
        "    sf: Scale factors\n"
        "    input: Input tensor (bfloat16)\n"
        "    global_scale: Global scale factor\n"
        "    cross_dim, reduce_dim: Dimension indices\n"
        "    swizzle: Enable memory swizzle");

  m.def("fp8_quantize", &ac4k::fp8_quantize_sm120,
        "Quantize bfloat16 tensor to FP8 format");

  m.def("int8_quantize", &ac4k::int8_quantize_sm120,
        "Quantize bfloat16 tensor to INT8 format");

  //-------------------------------------------------------------------------
  // GEMM / Dot Operators
  //-------------------------------------------------------------------------
  m.def("nvfp4_dot_scale", &ac4k::nvfp4_dot_scale_sm120,
        "NVFP4 Dot Product with Scaling\n\n"
        "D = A @ B.T with block-wise scaling");

  //-------------------------------------------------------------------------
  // Position Encoding Operators
  //-------------------------------------------------------------------------
  m.def("rope_3d_apply", &ac4k::rope_3d_apply,
        "Apply 3D Rotary Position Embedding\n\n"
        "Args:\n"
        "    x: Input tensor [B, S, N, D], bfloat16\n"
        "    grid_sizes: Grid sizes [B, 3], int32\n"
        "    freqs: Frequency table [max_pos, C], complex128\n"
        "    output: Output tensor [B, S, N, D], bfloat16");

  //-------------------------------------------------------------------------
  // Legacy API (with explicit arch suffix, for compatibility)
  //-------------------------------------------------------------------------
  m.def("nvfp4_mha_fwd_sm120", &ac4k::nvfp4_mha_fwd_sm120,
        "[Legacy] Use nvfp4_mha_fwd instead");
  m.def("qk_int8_pv_fp8_mha_fwd_sm120", &ac4k::qk_int8_pv_fp8_mha_fwd_sm120,
        "[Legacy] Use qk_int8_pv_fp8_mha_fwd instead");
  m.def("nvfp4_quantize_sm120", &ac4k::nvfp4_quantize_sm120,
        "[Legacy] Use nvfp4_quantize instead");
  m.def("fp8_quantize_sm120", &ac4k::fp8_quantize_sm120,
        "[Legacy] Use fp8_quantize instead");
  m.def("int8_quantize_sm120", &ac4k::int8_quantize_sm120,
        "[Legacy] Use int8_quantize instead");
  m.def("nvfp4_dot_scale_sm120", &ac4k::nvfp4_dot_scale_sm120,
        "[Legacy] Use nvfp4_dot_scale instead");
}
