#include <ATen/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <numeric>
#include <sstream>
#include <torch/all.h>
#include <vector>

#include "ac4k_kernel/ops/cuda_ops.h"
#include "utils.cuh"

#define CHECK_TYPE(x, st, m)                                                   \
  TORCH_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type:", m)
#define CHECK_TH_CUDA(x, m) TORCH_CHECK(x.is_cuda(), m, "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m)                                                 \
  TORCH_CHECK(x.is_contiguous(), m, "must be contiguous")
#define CHECK_INPUT(x, st, m)                                                  \
  CHECK_TH_CUDA(x, m);                                                         \
  CHECK_CONTIGUOUS(x, m);                                                      \
  CHECK_TYPE(x, st, m)
#define CHECK_OUTPUT(x, st, m) CHECK_INPUT(x, st, m)

namespace ac4k {

using BF16 = __nv_bfloat16;
using NVFP4x2 = uint8_t;
using NVFP4x8 = uint32_t;
using E4M3 = uint8_t;

constexpr int BLOCK_SIZE = 16;
constexpr int PACK_SF = sizeof(uint32_t) / sizeof(uint8_t);
constexpr int TILE_BLOCK_QUANTIZE_DIM = 64;
constexpr int TILE_BLOCK_NON_QUANTIZE_DIM = 4;
constexpr int TILE_THREAD_QUANTIZE_DIM = 8;
constexpr int TILE_THREAD_NON_QUANTIZE_DIM = 1;
constexpr int GROUP_SIZE = 32;
constexpr int THREAD_NUM_PER_GROUP = GROUP_SIZE / TILE_THREAD_QUANTIZE_DIM;

__forceinline__ __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

__forceinline__ __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
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
}

/// Reorder out
/// TODO
/// reshape: [quantize_dim] -> [-1, 4, 4, 2]
/// transpose: [0, 2, 1, 3]
///
/// SF layout
/// [quantize_dim/64, dim0 * dim1 * dim2, 4]xE4M3
///
/// OUT layout
/// [non_quantize_dim0, non_quantize_dim1, non_quantize_dim2,
///  ceil_up(quantize_dim, 64)/2]xnvfp4x2

template <bool Swizzle>
__global__ void
quantize_sm120_kernel(NVFP4x2 *out, E4M3 *sf, const BF16 *in,
                      const float *rcp_global_scale, int64_t non_quantize_dim0,
                      int64_t non_quantize_dim1, int64_t non_quantize_dim2,
                      int64_t quantize_dim, int64_t non_quantize_stride0,
                      int64_t non_quantize_stride1,
                      int64_t non_quantize_stride2, int64_t quantize_stride) {
  /// SF stride
  int64_t sf_stride2 = 1;
  int64_t sf_stride1 = PACK_SF * sf_stride2;
  int64_t sf_stride0 =
      non_quantize_dim0 * non_quantize_dim1 * non_quantize_dim2 * sf_stride1;
  /// OUT stride
  int64_t out_stride3 = 1;
  int64_t out_stride2 =
      align_up(quantize_dim, BLOCK_SIZE * PACK_SF) / 2 * out_stride3;
  int64_t out_stride1 = non_quantize_dim2 * out_stride2;
  int64_t out_stride0 = non_quantize_dim1 * out_stride1;

  float alpha = *rcp_global_scale;

  for (int dim0 = 0; dim0 < non_quantize_dim0; ++dim0) {
    int dim1 = blockIdx.z;
    int dim2 = (blockIdx.y * TILE_BLOCK_NON_QUANTIZE_DIM +
                threadIdx.y * TILE_THREAD_NON_QUANTIZE_DIM) %
               non_quantize_dim2;
    int apply_dim = blockIdx.x * TILE_BLOCK_QUANTIZE_DIM +
                    threadIdx.x * TILE_THREAD_QUANTIZE_DIM;

    BF16 bf16[TILE_THREAD_QUANTIZE_DIM];
    /// TODO: need opt load
    if constexpr (Swizzle) {
#pragma unroll
      for (int i = 0; i < TILE_THREAD_QUANTIZE_DIM; ++i) {
        int group_id = threadIdx.x / THREAD_NUM_PER_GROUP;
        int tid_in_group = threadIdx.x % THREAD_NUM_PER_GROUP;
        int apply_dim = blockIdx.x * TILE_BLOCK_QUANTIZE_DIM +
                        group_id * GROUP_SIZE + 2 * tid_in_group + i / 2 * 8 +
                        (i % 2);
        if (apply_dim < quantize_dim) {
          bf16[i] =
              in[dim0 * non_quantize_stride0 + dim1 * non_quantize_stride1 +
                 dim2 * non_quantize_stride2 + apply_dim * quantize_stride];
        } else {
          reinterpret_cast<uint16_t *>(bf16)[i] = 0;
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < TILE_THREAD_QUANTIZE_DIM; ++i) {
        if (apply_dim + i < quantize_dim) {
          bf16[i] =
              in[dim0 * non_quantize_stride0 + dim1 * non_quantize_stride1 +
                 dim2 * non_quantize_stride2 +
                 (apply_dim + i) * quantize_stride];
        } else {
          reinterpret_cast<uint16_t *>(bf16)[i] = 0;
        }
      }
    }

    __nv_bfloat162 *bf16x2 = reinterpret_cast<__nv_bfloat162 *>(bf16);
    // mas(abs)
    auto max_local = __habs2(bf16x2[0]);
#pragma unroll
    for (int i = 1; i < TILE_THREAD_QUANTIZE_DIM / 2; ++i) {
      max_local = __hmax2(max_local, __habs2(bf16x2[i]));
    }
    max_local = __hmax2(__shfl_xor_sync(0xffffffff, max_local, 1), max_local);

    /// max(abs)
    float max = __bfloat162float(__hmax(max_local.x, max_local.y));

    /// SF
    float sf_value = alpha * (max * 0.16666666666666666f);
    uint8_t sfu8;
    {
      __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(sf_value);
      sfu8 = static_cast<uint16_t>(tmp.__x);
      sf_value = static_cast<float>(tmp);
    }
    if (threadIdx.x % 2 == 0) {
      sf[apply_dim / 64 * sf_stride0 +
         dim0 * non_quantize_dim1 * non_quantize_dim2 * 4 +
         dim1 * non_quantize_dim2 * 4 + dim2 * 4 + threadIdx.x / 2] = sfu8;
    }

    float out_scale =
        sf_value != 0 ? alpha * reciprocal_approximate_ftz(sf_value) : 0.0f;

    float2 f32x2[TILE_THREAD_QUANTIZE_DIM / 2];
#pragma unroll
    for (int i = 0; i < TILE_THREAD_QUANTIZE_DIM / 2; i++) {
      f32x2[i] = __bfloat1622float2(bf16x2[i]);
      f32x2[i].x *= out_scale;
      f32x2[i].y *= out_scale;
    }

    NVFP4x8 e2m1x8 = fp32_vec_to_e2m1(f32x2);
    *reinterpret_cast<NVFP4x8 *>(out + dim0 * out_stride0 + dim1 * out_stride1 +
                                 dim2 * out_stride2 + apply_dim / 2) = e2m1x8;
  }
}

/// OUT layout
/// [non_dim0, non_dim1, non_dim2, ceil_up(quantize_dim, 64)/2]xNVFP4x2
///
/// SF layout
/// [ceil_div(quantize_dim, 64), non_dim0 * non_dim1 * non_dim2, 4]xE4M3
void quantize_sm120(torch::Tensor &out, torch::Tensor &sf,
                    torch::Tensor const &in,
                    torch::Tensor const &rcp_global_scale, uint32_t dim,
                    bool swizzle) {
  /// Check in
  std::vector<int64_t> in_shape;
  std::vector<int64_t> in_stride;
  CHECK_INPUT(in, at::ScalarType::BFloat16, "in must be bfloat16");
  TORCH_CHECK(in.dim() <= 4 && in.dim() >= 1, "in must be 1-4D");
  for (int i = 0; i < in.dim(); i++) {
    in_shape.push_back(in.size(i));
    in_stride.push_back(in.stride(i));
  }
  /// Check rcp_global_scale
  CHECK_INPUT(rcp_global_scale, at::ScalarType::Float,
              "rcp_global_scale must be float");
  TORCH_CHECK(rcp_global_scale.dim() == 0, "rcp_global_scale must be a scalar");

  /// Check dim
  TORCH_CHECK(dim < static_cast<uint32_t>(in.dim()),
              "dim must be less than in.dim()");
  int64_t quantize_dim_size = in_shape[dim];

  /// Check out
  std::vector<int64_t> out_shape;
  CHECK_OUTPUT(out, at::ScalarType::Byte, "out must be pack to uint8 tensor");
  TORCH_CHECK(out.dim() == in_shape.size(), "out must be the same dim as in");
  for (int i = 0; i < out.dim(); ++i) {
    out_shape.push_back(out.size(i));
  }
  for (uint32_t i = 0, counter = 0; i < in_shape.size(); ++i) {
    if (i != dim) {
      TORCH_CHECK(in_shape[i] == out_shape[counter++],
                  "out must be the same shape as in");
    }
  }
  TORCH_CHECK(out_shape[in_shape.size() - 1] * 2 ==
                  align_up(quantize_dim_size, BLOCK_SIZE * PACK_SF),
              "meet invalid quantize dim size");

  /// Check sf
  CHECK_OUTPUT(sf, at::ScalarType::Float8_e4m3fn, "sf must be float8 tensor");
  /// sf layout: [quantize_dim_size / 64, non_quantize_dim_size, 4]xfp8
  TORCH_CHECK(sf.dim() == 3, "sf must be 3D");
  TORCH_CHECK(sf.size(0) == ceil_div(quantize_dim_size, BLOCK_SIZE * PACK_SF),
              "meet invalid sf shape");
  TORCH_CHECK(sf.size(1) == std::accumulate(in_shape.begin(), in_shape.end(),
                                            1L, std::multiplies<int64_t>()) /
                                quantize_dim_size,
              "meet invalid sf shape");
  TORCH_CHECK(sf.size(2) == PACK_SF, "meet invalid sf shape");

  /// Expand to 4D
  for (int i = in.dim(); i < 4; ++i) {
    in_shape.insert(in_shape.begin(), 1);
    in_stride.insert(in_stride.begin(), in_shape[1] * in_stride[0]);
    out_shape.insert(out_shape.begin(), 1);
  }
  dim += in_shape.size() - in.dim();

  auto get_non_quantize_stride = [&](int i) -> int64_t {
    if (i < dim) {
      return in_stride[i];
    } else {
      return in_stride[i + 1];
    }
  };

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  dim3 grid(ceil_div(quantize_dim_size, TILE_BLOCK_QUANTIZE_DIM),
            ceil_div(out_shape[2], TILE_BLOCK_NON_QUANTIZE_DIM), out_shape[1]);
  dim3 block(TILE_BLOCK_QUANTIZE_DIM / TILE_THREAD_QUANTIZE_DIM,
             TILE_BLOCK_NON_QUANTIZE_DIM / TILE_THREAD_NON_QUANTIZE_DIM);
  if (swizzle) {
    quantize_sm120_kernel<true><<<grid, block, 0, stream>>>(
        reinterpret_cast<NVFP4x2 *>(out.data_ptr()),
        reinterpret_cast<E4M3 *>(sf.data_ptr()),
        reinterpret_cast<const BF16 *>(in.data_ptr()),
        reinterpret_cast<const float *>(rcp_global_scale.data_ptr()),
        out_shape[0], out_shape[1], out_shape[2], quantize_dim_size,
        get_non_quantize_stride(0), get_non_quantize_stride(1),
        get_non_quantize_stride(2), in_stride[dim]);
  } else {
    quantize_sm120_kernel<false><<<grid, block, 0, stream>>>(
        reinterpret_cast<NVFP4x2 *>(out.data_ptr()),
        reinterpret_cast<E4M3 *>(sf.data_ptr()),
        reinterpret_cast<const BF16 *>(in.data_ptr()),
        reinterpret_cast<const float *>(rcp_global_scale.data_ptr()),
        out_shape[0], out_shape[1], out_shape[2], quantize_dim_size,
        get_non_quantize_stride(0), get_non_quantize_stride(1),
        get_non_quantize_stride(2), in_stride[dim]);
  }
}

} // namespace ac4k
