#include <pybind11/pybind11.h>

#include "ac4k_kernel/ops/cuda_ops.h"

//===----------------------------------------------------------------------===//
// Register python module: module name "_cuda_ops" (internal use)
//===----------------------------------------------------------------------===//

PYBIND11_MODULE(_cuda_ops, m) {
  /// Matmul
  m.def("nvfp4_matmul_sm120", &ac4k::nvfp4_matmul_sm120,
        "CUDA-accelerated nvfp4_matmul_sm120");
  m.def("_internal_nvfp4_matmul_sm120", &ac4k::_internal_nvfp4_matmul_sm120,
        "Internal testing function");

  /// Quantization
  m.def("nvfp4_quant_sm120", &ac4k::nvfp4_quant_sm120,
        "CUDA-accelerated nvfp4_quant_sm120");
}
