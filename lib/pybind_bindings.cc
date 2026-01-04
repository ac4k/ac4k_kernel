#include <pybind11/pybind11.h>

#include "ac4k_kernel/ops/cuda_ops.h"

//===----------------------------------------------------------------------===//
// Register python module: module name "_cuda_ops" (internal use)
//===----------------------------------------------------------------------===//

PYBIND11_MODULE(_cuda_ops, m) {
  /// Dot
  m.def("nvfp4_dot_scale_sm120", &ac4k::nvfp4_dot_scale_sm120,
        "CUDA-accelerated nvfp4_dot_scale_sm120.");

  /// MHA forward
  m.def("nvfp4_mha_fwd_sm120", &ac4k::nvfp4_mha_fwd_sm120,
        "CUDA-accelerated mha fwd with nvfp4 precision.");
  m.def("qk_int8_pv_fp8_mha_fwd_sm120", &ac4k::qk_int8_pv_fp8_mha_fwd_sm120,
        "CUDA-accelerated mha fwd with qk int8 and pv fp8 precision.");

  /// Quantize
  m.def("nvfp4_quantize_sm120", &ac4k::nvfp4_quantize_sm120,
        "CUDA-accelerated bfloat16 to nvfp4 quantize.");
  m.def("fp8_quantize_sm120", &ac4k::fp8_quantize_sm120,
        "CUDA-accelerated fp8_quantize_sm120.");
  m.def("int8_quantize_sm120", &ac4k::int8_quantize_sm120,
        "CUDA-accelerated int8_quantize_sm120.");

  /// RoPE 3D apply
  m.def("rope_3d_apply", &ac4k::rope_3d_apply,
        "CUDA-accelerated 3D Rotary Position Embedding.");
}
