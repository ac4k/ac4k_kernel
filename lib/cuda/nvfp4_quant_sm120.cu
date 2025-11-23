#include <ATen/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sstream>
#include <torch/all.h>

#include "ac4k_kernel/ops/cuda_ops.h"
#include "utils.cuh"

namespace ac4k {

namespace {

#ifndef USE_ROCM
// Adapt from FlashInfer
#ifdef FLASHINFER_ENABLE_F16
#define _DISPATCH_CASE_F16(c_type, ...)                                        \
  case at::ScalarType::Half: {                                                 \
    using c_type = nv_half;                                                    \
    return __VA_ARGS__();                                                      \
  }
#else
#define _DISPATCH_CASE_F16(c_type, ...)
#endif

#ifdef FLASHINFER_ENABLE_BF16
#define _DISPATCH_CASE_BF16(c_type, ...)                                       \
  case at::ScalarType::BFloat16: {                                             \
    using c_type = nv_bfloat16;                                                \
    return __VA_ARGS__();                                                      \
  }
#else
#define _DISPATCH_CASE_BF16(c_type, ...)
#endif

#ifdef FLASHINFER_ENABLE_FP8_E4M3
#define _DISPATCH_CASE_FP8_E4M3(c_type, ...)                                   \
  case at::ScalarType::Float8_e4m3fn: {                                        \
    using c_type = __nv_fp8_e4m3;                                              \
    return __VA_ARGS__();                                                      \
  }
#else
#define _DISPATCH_CASE_FP8_E4M3(c_type, ...)
#endif

#ifdef FLASHINFER_ENABLE_FP8_E5M2
#define _DISPATCH_CASE_FP8_E5M2(c_type, ...)                                   \
  case at::ScalarType::Float8_e5m2: {                                          \
    using c_type = __nv_fp8_e5m2;                                              \
    return __VA_ARGS__();                                                      \
  }
#else
#define _DISPATCH_CASE_FP8_E5M2(c_type, ...)
#endif

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(pytorch_dtype, c_type, ...)       \
  [&]() -> bool {                                                              \
    switch (pytorch_dtype) {                                                   \
      _DISPATCH_CASE_F16(c_type, __VA_ARGS__)                                  \
      _DISPATCH_CASE_BF16(c_type, __VA_ARGS__)                                 \
    default:                                                                   \
      std::ostringstream oss;                                                  \
      oss << __PRETTY_FUNCTION__ << " failed to dispatch data type "           \
          << pytorch_dtype;                                                    \
      TORCH_CHECK(false, oss.str());                                           \
      return false;                                                            \
    }                                                                          \
  }()

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(pytorch_dtype, c_type, ...)        \
  [&]() -> bool {                                                              \
    switch (pytorch_dtype) {                                                   \
      _DISPATCH_CASE_FP8_E4M3(c_type, __VA_ARGS__)                             \
      _DISPATCH_CASE_FP8_E5M2(c_type, __VA_ARGS__)                             \
    default:                                                                   \
      std::ostringstream oss;                                                  \
      oss << __PRETTY_FUNCTION__ << " failed to dispatch fp8 data type "       \
          << pytorch_dtype;                                                    \
      TORCH_CHECK(false, oss.str());                                           \
      return false;                                                            \
    }                                                                          \
  }()

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE(pytorch_dtype, c_type, ...)            \
  [&]() -> bool {                                                              \
    switch (pytorch_dtype) {                                                   \
      _DISPATCH_CASE_F16(c_type, __VA_ARGS__)                                  \
      _DISPATCH_CASE_BF16(c_type, __VA_ARGS__)                                 \
      _DISPATCH_CASE_FP8_E4M3(c_type, __VA_ARGS__)                             \
      _DISPATCH_CASE_FP8_E5M2(c_type, __VA_ARGS__)                             \
    default:                                                                   \
      std::ostringstream oss;                                                  \
      oss << __PRETTY_FUNCTION__ << " failed to dispatch data type "           \
          << pytorch_dtype;                                                    \
      TORCH_CHECK(false, oss.str());                                           \
      return false;                                                            \
    }                                                                          \
  }()

#define _DISPATCH_SWITCH(var_name, cond, ...)                                  \
  [&]() -> bool {                                                              \
    switch (cond) {                                                            \
      __VA_ARGS__                                                              \
    default:                                                                   \
      std::ostringstream oss;                                                  \
      oss << __PRETTY_FUNCTION__ << " failed to dispatch " var_name " "        \
          << int(cond);                                                        \
      TORCH_CHECK(false, oss.str());                                           \
      return false;                                                            \
    }                                                                          \
  }()

#define _DISPATCH_SWITCH_U16x2(var1_name, var2_name, cond1, cond2, ...)        \
  [&]() -> bool {                                                              \
    switch (pack_u16(cond1, cond2)) {                                          \
      __VA_ARGS__                                                              \
    default:                                                                   \
      std::ostringstream oss;                                                  \
      oss << __PRETTY_FUNCTION__                                               \
          << " failed to dispatch (" var1_name ", " var2_name "): ("           \
          << int(cond1) << ", " << int(cond2) << ")";                          \
      TORCH_CHECK(false, oss.str());                                           \
      return false;                                                            \
    }                                                                          \
  }()

#define _DISPATCH_CASE(case_expr, case_var, ...)                               \
  case case_expr: {                                                            \
    constexpr auto case_var = case_expr;                                       \
    return __VA_ARGS__();                                                      \
  }

#define _DISPATCH_CASE_U16x2(case_expr1, case_expr2, case_var1, case_var2,     \
                             ...)                                              \
  case pack_u16(case_expr1, case_expr2): {                                     \
    constexpr auto case_var1 = case_expr1;                                     \
    constexpr auto case_var2 = case_expr2;                                     \
    return __VA_ARGS__();                                                      \
  }

#define DISPATCH_BOOL(expr, const_expr, ...)                                   \
  [&]() -> bool {                                                              \
    if (expr) {                                                                \
      constexpr bool const_expr = true;                                        \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr bool const_expr = false;                                       \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

inline void check_shape(const torch::Tensor &a, const torch::Tensor &b,
                        const char *a_name, const char *b_name) {
  TORCH_CHECK(a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ",
              a.dim(), " vs ", b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name,
                ".size(", i, ")");
  }
}

inline constexpr uint32_t pack_u16(uint16_t a, uint16_t b) {
  return (uint32_t(a) << 16) | uint32_t(b);
}

#define CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads)                   \
  TORCH_CHECK(num_qo_heads % num_kv_heads == 0, "num_qo_heads(", num_qo_heads, \
              ") must be divisible by num_kv_heads(", num_kv_heads, ")")

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LAST_DIM_CONTIGUOUS(x)                                           \
  TORCH_CHECK(x.strides()[x.strides().size() - 1] == 1,                        \
              #x "must be contiguous at last dimension")

#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)
#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x)                                     \
  CHECK_CUDA(x);                                                               \
  CHECK_LAST_DIM_CONTIGUOUS(x)

#define CHECK_DIM(d, x)                                                        \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b)                                                         \
  TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#define CHECK_GE(a, b)                                                         \
  TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed. ", a, " vs ", b)

inline bool is_float8_tensor(const torch::Tensor &tensor) {
  return tensor.scalar_type() == at::ScalarType::Float8_e4m3fn ||
         tensor.scalar_type() == at::ScalarType::Float8_e5m2;
}
#endif

struct cuda_error : public std::runtime_error {
  /**
   * @brief Constructs a `cuda_error` object with the given `message`.
   *
   * @param message The error char array used to construct `cuda_error`
   */
  cuda_error(const char *message) : std::runtime_error(message) {}
  /**
   * @brief Constructs a `cuda_error` object with the given `message` string.
   *
   * @param message The `std::string` used to construct `cuda_error`
   */
  cuda_error(std::string const &message) : cuda_error{message.c_str()} {}
};

#define CHECK_CUDA_SUCCESS(cmd)                                                \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      std::stringstream _message;                                              \
      auto s = cudaGetErrorString(e);                                          \
      _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;        \
      throw cuda_error(_message.str());                                        \
    }                                                                          \
  } while (0)

#define CHECK_IS_CUDA(x)                                                       \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_IS_CONTIGUOUS(x)                                                 \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_INPUT(x)                                                    \
  CHECK_IS_CUDA(x);                                                            \
  CHECK_IS_CONTIGUOUS(x)

inline int getSMVersion() {
  int device{-1};
  CHECK_CUDA_SUCCESS(cudaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(
      &sm_major, cudaDevAttrComputeCapabilityMajor, device));
  CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(
      &sm_minor, cudaDevAttrComputeCapabilityMinor, device));
  return sm_major * 10 + sm_minor;
}

// SGLANG_SHFL_XOR_* adapted from
// https://github.com/vllm-project/vllm/blob/v0.7.3/csrc/cuda_compat.h#L19-L28
#ifndef USE_ROCM
#define SGLANG_SHFL_XOR_SYNC(mask, var, lane_mask)                             \
  __shfl_xor_sync((mask), (var), (lane_mask))
#define SGLANG_SHFL_XOR_SYNC_WIDTH(mask, var, lane_mask, width)                \
  __shfl_xor_sync((mask), (var), (lane_mask), (width))
#else
#define SGLANG_SHFL_XOR_SYNC(mask, var, lane_mask)                             \
  __shfl_xor((var), (lane_mask))
#define SGLANG_SHFL_XOR_SYNC_WIDTH(mask, var, lane_mask, width)                \
  __shfl_xor((var), (lane_mask), (width))
#endif

#ifndef USE_ROCM
#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(pytorch_dtype, c_type, ...) \
  [&]() -> bool {                                                              \
    switch (pytorch_dtype) {                                                   \
    case at::ScalarType::Float: {                                              \
      using c_type = float;                                                    \
      return __VA_ARGS__();                                                    \
    }                                                                          \
      _DISPATCH_CASE_F16(c_type, __VA_ARGS__)                                  \
      _DISPATCH_CASE_BF16(c_type, __VA_ARGS__)                                 \
    default:                                                                   \
      std::ostringstream oss;                                                  \
      oss << __PRETTY_FUNCTION__ << " failed to dispatch data type "           \
          << pytorch_dtype;                                                    \
      TORCH_CHECK(false, oss.str());                                           \
      return false;                                                            \
    }                                                                          \
  }()
#endif

#define DISPATCH_CASE_INTEGRAL_TYPES(...)                                      \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)                          \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)                          \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__)                         \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)                           \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...)                               \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#define WARP_SIZE 32

#ifndef USE_ROCM
#include <c10/util/Float8_e4m3fn.h>
using FP8_TYPE = c10::Float8_e4m3fn;
C10_HOST_DEVICE constexpr auto FP8_E4M3_MAX =
    std::numeric_limits<FP8_TYPE>::max();
#else
#include <c10/util/Float8_e4m3fnuz.h>

using FP8_TYPE = c10::Float8_e4m3fnuz;
constexpr auto FP8_E4M3_MAX = 224.0f;
#endif

#ifndef USE_ROCM
__device__ __forceinline__ float atomicMaxFloat(float *addr, float value) {
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int *)addr, __float_as_uint(value)));
  return old;
}

__device__ __forceinline__ float warpReduceMax(float max_value) {
  max_value = fmaxf(max_value, SGLANG_SHFL_XOR_SYNC(0xffffffff, max_value, 16));
  max_value = fmaxf(max_value, SGLANG_SHFL_XOR_SYNC(0xffffffff, max_value, 8));
  max_value = fmaxf(max_value, SGLANG_SHFL_XOR_SYNC(0xffffffff, max_value, 4));
  max_value = fmaxf(max_value, SGLANG_SHFL_XOR_SYNC(0xffffffff, max_value, 2));
  max_value = fmaxf(max_value, SGLANG_SHFL_XOR_SYNC(0xffffffff, max_value, 1));
  return max_value;
}

__device__ __forceinline__ float blockReduceMax(float max_value) {
  static __shared__ float warpLevelMaxs[WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;

  max_value = warpReduceMax(max_value);

  if (laneId == 0)
    warpLevelMaxs[warpId] = max_value;
  __syncthreads();

  max_value =
      (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelMaxs[laneId] : 0;
  if (warpId == 0)
    max_value = warpReduceMax(max_value);

  return max_value;
}
#endif

// Pads to a multiple of `alignment` rows.
inline torch::Tensor pad_tensor(const torch::Tensor &tensor,
                                int64_t alignment = 4,
                                bool is_column_major = false) {
  int64_t rows = tensor.size(0);
  int64_t cols = tensor.size(1);
  int64_t pad_rows =
      (alignment - (rows % alignment)) % alignment; // Compute padding size

  if (pad_rows == 0) {
    return tensor; // Already aligned
  }

  torch::Tensor padding = at::zeros({pad_rows, cols}, tensor.options());
  torch::Tensor tensor_padded = at::cat({tensor, padding}, 0); // Pad along rows

  // Ensure column-major layout
  if (is_column_major) {
    return tensor_padded.t().contiguous().t();
  }
  return tensor_padded;
}

} // namespace

// Get type2 from type or vice versa (applied to half and bfloat16)
template <typename T> struct TypeConverter {
  using Type = half2;
}; // keep for generality

template <> struct TypeConverter<half2> {
  using Type = half;
};

template <> struct TypeConverter<half> {
  using Type = half2;
};

template <> struct TypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};

template <> struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

#define ELTS_PER_THREAD 8

constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
constexpr int CVT_FP4_SF_VEC_SIZE = 16;

// Convert 8 float32 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
  // PTX instructions used here requires sm100a.
  // #if CUDA_VERSION >= 12080
  // #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) &&
  // __CUDA_ARCH_HAS_FEATURE__(SM100_ALL)
  uint32_t val;
  asm volatile("{\n"
               ".reg .b8 byte0;\n"
               ".reg .b8 byte1;\n"
               ".reg .b8 byte2;\n"
               ".reg .b8 byte3;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
               "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
               "}"
               : "=r"(val)
               : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]),
                 "f"(array[4]), "f"(array[5]), "f"(array[6]), "f"(array[7]));
  return val;
  // #else
  //   return 0;
  // #endif
  // #endif
}

// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
  // PTX instructions used here requires sm100a.
  // #if CUDA_VERSION >= 12080
  // #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) &&
  // __CUDA_ARCH_HAS_FEATURE__(SM100_ALL)
  uint32_t val;
  asm volatile("{\n"
               ".reg .b8 byte0;\n"
               ".reg .b8 byte1;\n"
               ".reg .b8 byte2;\n"
               ".reg .b8 byte3;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
               "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
               "}"
               : "=r"(val)
               : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x),
                 "f"(array[1].y), "f"(array[2].x), "f"(array[2].y),
                 "f"(array[3].x), "f"(array[3].y));
  return val;
  // #else
  //   return 0;
  // #endif
  // #endif
}

// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t *cvt_quant_to_fp4_get_sf_out_offset(int rowIdx, int colIdx,
                                                       int numCols,
                                                       SFType *SFout) {
  // #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 ||
                CVT_FP4_NUM_THREADS_PER_SF == 2);

  // One pair of threads write one SF to global memory.
  // TODO: stage through smem for packed STG.32
  // is it better than STG.8 from 4 threads ?
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    // SF vector index (16 elements share one SF in the K dimension).
    int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;

    // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
    // --> index [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

    int32_t mTileIdx = mIdx / (32 * 4);
    // SF vector size 16.
    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    int32_t numKTiles = (numCols + factor - 1) / factor;
    int64_t mTileStride = numKTiles * 32 * 4 * 4;

    int32_t kTileIdx = (kIdx / 4);
    int64_t kTileStride = 32 * 4 * 4;

    // M tile layout [32, 4] is column-major.
    int32_t outerMIdx = (mIdx % 32);
    int64_t outerMStride = 4 * 4;

    int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
    int64_t innerMStride = 4;

    int32_t innerKIdx = (kIdx % 4);
    int64_t innerKStride = 1;

    // Compute the global offset.
    int64_t SFOffset = mTileIdx * mTileStride + kTileIdx * kTileStride +
                       outerMIdx * outerMStride + innerMIdx * innerMStride +
                       innerKIdx * innerKStride;

    return reinterpret_cast<uint8_t *>(SFout) + SFOffset;
  }
  // #endif
  return nullptr;
}

// Define a 16 bytes packed data type.
template <class Type> struct PackedVec {
  typename TypeConverter<Type>::Type elts[4];
};

template <> struct PackedVec<__nv_fp8_e4m3> {
  __nv_fp8x2_e4m3 elts[8];
};

template <class T>
__device__ typename std::enable_if<std::is_same<T, __half>::value ||
                                       std::is_same<T, __nv_bfloat16>::value,
                                   float>::type
to_float(T val) {
  return 0.0f;
}
template <> __device__ float to_float<__nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}
template <> __device__ float to_float<__half>(__half val) {
  return __half2float(val);
}

// Quantizes the provided PackedVec into the uint32_t output
template <class Type, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec<Type> &vec, float SFScaleVal,
                                         uint8_t *SFout) {
  // #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  // Get absolute maximum values among the local 8 values.
  auto localMax = __habs2(vec.elts[0]);

// Local maximum value.
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  // Get the absolute maximum among all 16 values (two threads).
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  // Get the final absolute maximum values.
  float vecMax = to_float<Type>(__hmax(localMax.x, localMax.y));

  // Get the SF (max value of the vector / max value of e2m1).
  // maximum value of e2m1 = 6.0.
  // TODO: use half as compute data type.
  float SFValue = SFScaleVal * (vecMax * 0.16666666666666666f);
  // 8 bits representation of the SF.
  uint8_t fp8SFVal;
  // Write the SF to global memory (STG.8).
  if constexpr (UE8M0_SF) {
    __nv_fp8_e8m0 tmp;
    tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
    SFValue = static_cast<float>(tmp);
    fp8SFVal = tmp.__x;
  } else {
    // Here SFValue is always positive, so E4M3 is the same as UE4M3.
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    fp8SFVal = tmp.__x;
    SFValue = static_cast<float>(tmp);
  }
  // Get the output scale.
  // Recipe: final_scale = reciprocal(fp32(fp8(SFValue * SFScaleVal))) *
  //                       reciprocal(SFScaleVal))
  //   float outputScale =
  //       SFValue != 0 ? reciprocal_approximate_ftz(SFValue *
  //       reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;

  float outputScale =
      SFValue != 0 ? SFScaleVal * reciprocal_approximate_ftz(SFValue) : 0.0f;

  if (SFout) {
    // Write the SF to global memory (STG.8).
    *SFout = fp8SFVal;
  }

  // Convert the input to float.
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Convert to e2m1 values.
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // Write the e2m1 values to global memory.
  return e2m1Vec;
  // #else
  //   return 0;
  // #endif
}

// Use UE4M3 by default.
template <class Type, bool UE8M0_SF = false>
__global__ void
// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(256, 6) cvt_fp16_to_fp4(
    // #else
    // cvt_fp16_to_fp4(
    // #endif
    int32_t numRows, int32_t numCols, Type const *in, float const *SFScale,
    uint32_t *out, uint32_t *SFout) {
  // #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // Get the global scaling factor, which will be applied to the SF.
  // Note SFScale is the same as next GEMM's alpha, which is
  // (448.f / (Alpha_A / 6.f)).
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

  // Input tensor row/col loops.
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP4_ELTS_PER_THREAD;
         colIdx += blockDim.x) {
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const *>(in)[inOffset];
      // Get the output tensor offset.
      // Same as inOffset because 8 elements are packed into one uint32_t.
      int64_t outOffset = inOffset;
      auto &out_pos = out[outOffset];

      auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<
          uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(rowIdx, colIdx, numCols, SFout);

      out_pos =
          cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
    }
  }
  // #endif
}

template <typename T>
void invokeFP4Quantization(int m, int n, T const *input, float const *SFScale,
                           int64_t *output, int32_t *SFOuput, bool useUE8M0,
                           int multiProcessorCount, cudaStream_t stream) {
  // Grid, Block size.
  // Each thread converts 8 values.
  dim3 block(std::min(int(n / ELTS_PER_THREAD), 256));
  // Get number of blocks per SM (assume we can fully utilize the SM).
  int const numBlocksPerSM = 1536 / block.x;
  dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

  // Launch the cvt kernel.
  if (useUE8M0) {
    cvt_fp16_to_fp4<T, true><<<grid, block, 0, stream>>>(
        m, n, input, SFScale, reinterpret_cast<uint32_t *>(output),
        reinterpret_cast<uint32_t *>(SFOuput));
  } else {
    cvt_fp16_to_fp4<T, false><<<grid, block, 0, stream>>>(
        m, n, input, SFScale, reinterpret_cast<uint32_t *>(output),
        reinterpret_cast<uint32_t *>(SFOuput));
  }
}

// Instantiate the function.
template void invokeFP4Quantization(int m, int n, half const *input,
                                    float const *SFScale, int64_t *output,
                                    int32_t *SFOuput, bool useUE8M0,
                                    int multiProcessorCount,
                                    cudaStream_t stream);

template void invokeFP4Quantization(int m, int n, __nv_bfloat16 const *input,
                                    float const *SFScale, int64_t *output,
                                    int32_t *SFOuput, bool useUE8M0,
                                    int multiProcessorCount,
                                    cudaStream_t stream);

inline int getMultiProcessorCount() {
  static int multi_processor_count = []() {
    int device_id = 0;
    int count = 0;

    // Get the current CUDA device ID
    CHECK_CUDA_SUCCESS(cudaGetDevice(&device_id));

    // Get the number of multiprocessors for the current device
    CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(
        &count, cudaDevAttrMultiProcessorCount, device_id));

    return count; // Initialize the static variable
  }();

  return multi_processor_count; // Return the cached value on subsequent calls
}

void nvfp4_quant_sm120(torch::Tensor &output, torch::Tensor &output_sf,
                       torch::Tensor const &input,
                       torch::Tensor const &input_global_scale) {
  int32_t m = input.size(0);
  int32_t n = input.size(1);

  TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");

  int multiProcessorCount = getMultiProcessorCount();

  auto input_global_scale_ptr =
      static_cast<float const *>(input_global_scale.data_ptr());
  auto sf_out = static_cast<int32_t *>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t *>(output.data_ptr());
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(input.get_device());

  // We don't support e8m0 scales at this moment.
  bool useUE8M0 = false;

  switch (input.scalar_type()) {
  case at::kHalf: {
    auto input_ptr = reinterpret_cast<half const *>(input.data_ptr());
    invokeFP4Quantization(m, n, input_ptr, input_global_scale_ptr, output_ptr,
                          sf_out, useUE8M0, multiProcessorCount, stream);
    break;
  }
  case at::kBFloat16: {
    auto input_ptr = reinterpret_cast<__nv_bfloat16 const *>(input.data_ptr());
    invokeFP4Quantization(m, n, input_ptr, input_global_scale_ptr, output_ptr,
                          sf_out, useUE8M0, multiProcessorCount, stream);
    break;
  }
  default: {
    std::cerr << "Observing: " << input.scalar_type()
              << " for the input datatype which is invalid";
    throw std::runtime_error(
        "Unsupported input data type for quantize_to_fp4.");
  }
  }
}

} // namespace ac4k
