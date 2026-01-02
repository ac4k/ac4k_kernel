#pragma once

#include "math.cuh"
#include "mma.cuh"

#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>
#include <string>

//===----------------------------------------------------------------------===//
// Error - checking macro
//===----------------------------------------------------------------------===//

/// Check CUDA Runtime API
#define CHECK_CUDA_ERROR(call)                                                 \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Runtime error at " << __FILE__ << ":" << __LINE__     \
                << " - ";                                                      \
      std::cerr << cudaGetErrorString(err) << " (" << #call << ")"             \
                << std::endl;                                                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/// Check CUDA Driver API
#define CHECK_CUDA_DRIVER_ERROR(call)                                          \
  do {                                                                         \
    CUresult err = call;                                                       \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *errName = nullptr;                                           \
      const char *errDesc = nullptr;                                           \
      cuGetErrorName(err, &errName);                                           \
      cuGetErrorString(err, &errDesc);                                         \
      std::cerr << "CUDA Driver error at " << __FILE__ << ":" << __LINE__      \
                << " - ";                                                      \
      std::cerr << (errName ? errName : "Unknown") << ": "                     \
                << (errDesc ? errDesc : "No description") << " (" << #call     \
                << ")" << std::endl;                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/// Check cuBLAS API
#define CHECK_CUBLAS_ERROR(call)                                               \
  do {                                                                         \
    cublasStatus_t err = call;                                                 \
    if (err != CUBLAS_STATUS_SUCCESS) {                                        \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - "; \
      std::cerr << cublasGetStatusString(err) << " (" << #call << ")"          \
                << std::endl;                                                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

namespace ac4k {

//===----------------------------------------------------------------------===//
// A, B and D fragment description
//===----------------------------------------------------------------------===//

struct AFrag_NVFP4_16x64 {
  using REG_TYPE = uint32_t;
  static constexpr int THREADS_PER_CHUNK = 4;
  static constexpr int BITS_PER_ELEMENT = 4;
  static constexpr int REGISTERS_PER_THREAD = 4;
  static constexpr int ELES_PER_REGISTER = 32 / BITS_PER_ELEMENT;
  static constexpr int ELES_PER_THREAD =
      ELES_PER_REGISTER * REGISTERS_PER_THREAD;

  REG_TYPE data[REGISTERS_PER_THREAD];

  __forceinline__ __device__ size_t get_row_with_ele(int tid, int ele_id) {
    int group_id = tid / THREADS_PER_CHUNK;
    return group_id + 8 * ((ele_id / ELES_PER_REGISTER) % 2);
  }

  __forceinline__ __device__ size_t get_col_with_ele(int tid, int ele_id) {
    return ELES_PER_REGISTER * (tid % 4) + (ele_id % ELES_PER_REGISTER) +
           32 * (ele_id / 16);
  }

  __forceinline__ __device__ size_t get_row_with_reg(int tid, int reg_id) {
    int ele_id = reg_id * ELES_PER_REGISTER;
    return get_row_with_ele(tid, ele_id);
  }

  __forceinline__ __device__ size_t get_col_with_reg(int tid, int reg_id) {
    int ele_id = reg_id * ELES_PER_REGISTER;
    return get_col_with_ele(tid, ele_id);
  }
};

struct AFrag_FP8_16x32 {
  using REG_TYPE = uint32_t;
  static constexpr int THREADS_PER_CHUNK = 4;
  static constexpr int BITS_PER_ELEMENT = 8;
  static constexpr int REGISTERS_PER_THREAD = 4;
  static constexpr int ELES_PER_REGISTER = 32 / BITS_PER_ELEMENT;
  static constexpr int ELES_PER_THREAD =
      ELES_PER_REGISTER * REGISTERS_PER_THREAD;

  REG_TYPE data[REGISTERS_PER_THREAD];

  __forceinline__ __device__ size_t get_row_with_ele(int tid, int ele_id) {
    int group_id = tid / THREADS_PER_CHUNK;
    return group_id + 8 * ((ele_id / ELES_PER_REGISTER) % 2);
  }

  __forceinline__ __device__ size_t get_col_with_ele(int tid, int ele_id) {
    return ELES_PER_REGISTER * (tid % 4) + (ele_id % ELES_PER_REGISTER) +
           16 * (ele_id / 8);
  }

  __forceinline__ __device__ size_t get_row_with_reg(int tid, int reg_id) {
    int ele_id = reg_id * ELES_PER_REGISTER;
    return get_row_with_ele(tid, ele_id);
  }

  __forceinline__ __device__ size_t get_col_with_reg(int tid, int reg_id) {
    int ele_id = reg_id * ELES_PER_REGISTER;
    return get_col_with_ele(tid, ele_id);
  }
};

using AFrag_I8_16x32 = AFrag_FP8_16x32;

struct BFrag_NVFP4_64x8 {
  using REG_TYPE = uint32_t;
  static constexpr int THREADS_PER_CHUNK = 4;
  static constexpr int BITS_PER_ELEMENT = 4;
  static constexpr int REGISTERS_PER_THREAD = 2;
  static constexpr int ELES_PER_REGISTER = 32 / BITS_PER_ELEMENT;
  static constexpr int ELES_PER_THREAD =
      ELES_PER_REGISTER * REGISTERS_PER_THREAD;

  REG_TYPE data[REGISTERS_PER_THREAD];

  __forceinline__ __device__ size_t get_row_with_ele(int tid, int ele_id) {
    return (ele_id % ELES_PER_REGISTER) +
           (tid % THREADS_PER_CHUNK) * ELES_PER_REGISTER +
           (ele_id / ELES_PER_REGISTER) *
               (THREADS_PER_CHUNK * ELES_PER_REGISTER);
  }

  __forceinline__ __device__ size_t get_col_with_ele(int tid, int ele_id) {
    return tid / THREADS_PER_CHUNK;
  }

  __forceinline__ __device__ size_t get_row_with_reg(int tid, int reg_id) {
    int ele_id = reg_id * ELES_PER_REGISTER;
    return get_row_with_ele(tid, ele_id);
  }

  __forceinline__ __device__ size_t get_col_with_reg(int tid, int reg_id) {
    int ele_id = reg_id * ELES_PER_REGISTER;
    return get_col_with_ele(tid, ele_id);
  }
};

struct BFrag_FP8_32x8 {
  using REG_TYPE = uint32_t;
  static constexpr int THREADS_PER_CHUNK = 4;
  static constexpr int BITS_PER_ELEMENT = 8;
  static constexpr int REGISTERS_PER_THREAD = 2;
  static constexpr int ELES_PER_REGISTER = 32 / BITS_PER_ELEMENT;
  static constexpr int ELES_PER_THREAD =
      ELES_PER_REGISTER * REGISTERS_PER_THREAD;

  REG_TYPE data[REGISTERS_PER_THREAD];

  __forceinline__ __device__ size_t get_row_with_ele(int tid, int ele_id) {
    return (ele_id % 8) + (tid % 4) * 8;
  }

  __forceinline__ __device__ size_t get_col_with_ele(int tid, int ele_id) {
    return tid / THREADS_PER_CHUNK;
  }

  __forceinline__ __device__ size_t get_row_with_reg(int tid, int reg_id) {
    int ele_id = reg_id * ELES_PER_REGISTER;
    return get_row_with_ele(tid, ele_id);
  }

  __forceinline__ __device__ size_t get_col_with_reg(int tid, int reg_id) {
    int ele_id = reg_id * ELES_PER_REGISTER;
    return get_col_with_ele(tid, ele_id);
  }
};

using BFrag_I8_32x8 = BFrag_FP8_32x8;

struct DFrag_F32_16x8 {
  using REG_TYPE = float;
  static constexpr int THREADS_PER_CHUNK = 4;
  static constexpr int BITS_PER_ELEMENT = 32;
  static constexpr int REGISTERS_PER_THREAD = 4;
  static constexpr int ELES_PER_REGISTER = 32 / BITS_PER_ELEMENT;
  static constexpr int ELES_PER_THREAD =
      ELES_PER_REGISTER * REGISTERS_PER_THREAD;

  REG_TYPE data[REGISTERS_PER_THREAD];

  __forceinline__ __device__ size_t get_row_with_ele(int tid, int ele_id) {
    return (tid / THREADS_PER_CHUNK) + 8 * (ele_id / 2);
  }

  __forceinline__ __device__ size_t get_col_with_ele(int tid, int ele_id) {
    return 2 * (tid % 4) + (ele_id % 2);
  }

  __forceinline__ __device__ size_t get_row_with_reg(int tid, int reg_id) {
    int ele_id = reg_id * ELES_PER_REGISTER;
    return get_row_with_ele(tid, ele_id);
  }

  __forceinline__ __device__ size_t get_col_with_reg(int tid, int reg_id) {
    int ele_id = reg_id * ELES_PER_REGISTER;
    return get_col_with_ele(tid, ele_id);
  }
};

struct DFrag_I32_16x8 {
  using REG_TYPE = int32_t;
  static constexpr int THREADS_PER_CHUNK = 4;
  static constexpr int BITS_PER_ELEMENT = 32;
  static constexpr int REGISTERS_PER_THREAD = 4;
  static constexpr int ELES_PER_REGISTER = 32 / BITS_PER_ELEMENT;
  static constexpr int ELES_PER_THREAD =
      ELES_PER_REGISTER * REGISTERS_PER_THREAD;

  REG_TYPE data[REGISTERS_PER_THREAD];

  __forceinline__ __device__ size_t get_row_with_ele(int tid, int ele_id) {
    return (tid / THREADS_PER_CHUNK) + 8 * (ele_id / 2);
  }

  __forceinline__ __device__ size_t get_col_with_ele(int tid, int ele_id) {
    return 2 * (tid % 4) + (ele_id % 2);
  }

  __forceinline__ __device__ size_t get_row_with_reg(int tid, int reg_id) {
    int ele_id = reg_id * ELES_PER_REGISTER;
    return get_row_with_ele(tid, ele_id);
  }

  __forceinline__ __device__ size_t get_col_with_reg(int tid, int reg_id) {
    int ele_id = reg_id * ELES_PER_REGISTER;
    return get_col_with_ele(tid, ele_id);
  }
};

struct DFrag_F16_16x8 {
  using REG_TYPE = uint32_t;
  static constexpr int THREADS_PER_CHUNK = 4;
  static constexpr int BITS_PER_ELEMENT = 16;
  static constexpr int REGISTERS_PER_THREAD = 2;
  static constexpr int ELES_PER_REGISTER = 32 / BITS_PER_ELEMENT;
  static constexpr int ELES_PER_THREAD =
      ELES_PER_REGISTER * REGISTERS_PER_THREAD;
  REG_TYPE data[REGISTERS_PER_THREAD];

  __forceinline__ __device__ size_t get_row_with_ele(int tid, int ele_id) {
    return (tid / THREADS_PER_CHUNK) + 8 * (ele_id / 2);
  }

  __forceinline__ __device__ size_t get_col_with_ele(int tid, int ele_id) {
    return 2 * (tid % 4) + (ele_id % 2);
  }

  __forceinline__ __device__ size_t get_row_with_reg(int tid, int reg_id) {
    int ele_id = reg_id * ELES_PER_REGISTER;
    return get_row_with_ele(tid, ele_id);
  }

  __forceinline__ __device__ size_t get_col_with_reg(int tid, int reg_id) {
    int ele_id = reg_id * ELES_PER_REGISTER;
    return get_col_with_ele(tid, ele_id);
  }
};

} // namespace ac4k
