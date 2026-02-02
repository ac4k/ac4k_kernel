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

#include "ac4k_kernel/ops.h"
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
constexpr int TILE_BLOCK_NON_QUANTIZE_DIM = 16;
constexpr int TILE_THREAD_QUANTIZE_DIM = 8;
constexpr int TILE_THREAD_NON_QUANTIZE_DIM = 1;
constexpr int GROUP_SIZE = 32;
constexpr int THREAD_NUM_PER_GROUP = GROUP_SIZE / TILE_THREAD_QUANTIZE_DIM;

constexpr int CROSS_DIM_ALIGN_SIZE = 16;
constexpr int REDUCE_DIM_ALIGN_SIZE = BLOCK_SIZE * PACK_SF;

template <bool Swizzle>
__global__ void nvfp4_quantize_sm120_kernel(
    NVFP4x2 *out, uint32_t *sf, const BF16 *in, const float *global_scale,
    int64_t in_dim0, int64_t in_dim1, int64_t /* cross-dim-size */ in_dim2,
    int64_t /* reduce-dim-size */ in_dim3, int64_t in_stride0,
    int64_t in_stride1, int64_t in_stride2, int64_t in_stride3) {
  /// SF stride(uint32_t)
  /// [xx, xx, reduce_dim_align / 64, cross_dim_align] x uint32
  int64_t sf_stride3 = 1;
  int64_t sf_stride2 = sf_stride3 * align_up(in_dim2, CROSS_DIM_ALIGN_SIZE);
  int64_t sf_stride1 = sf_stride2 * ceil_div(in_dim3, REDUCE_DIM_ALIGN_SIZE);
  int64_t sf_stride0 = sf_stride1 * in_dim1;

  /// OUT stride
  /// [xx, xx, cross_dim_align, reduce_dim_align / 2] x NVFP4x2
  int64_t out_dim0 = in_dim0;
  int64_t out_dim1 = in_dim1;
  int64_t out_dim2 = in_dim2;
  int64_t out_dim3 = align_up(in_dim3, REDUCE_DIM_ALIGN_SIZE) / 2;
  int64_t out_stride3 = 1;
  int64_t out_stride2 = out_stride3 * out_dim3;
  int64_t out_stride1 = out_stride2 * out_dim2;
  int64_t out_stride0 = out_stride1 * out_dim1;

  float alpha = reciprocal_approximate_ftz(*global_scale);

  for (int dim0 = 0; dim0 < out_dim0; ++dim0) {
    int dim1 = blockIdx.z;
    int dim2 = blockIdx.y * TILE_BLOCK_NON_QUANTIZE_DIM +
               threadIdx.y * TILE_THREAD_NON_QUANTIZE_DIM;
    int dim3 = blockIdx.x * TILE_BLOCK_QUANTIZE_DIM +
               threadIdx.x * TILE_THREAD_QUANTIZE_DIM;

    BF16 bf16[TILE_THREAD_QUANTIZE_DIM];
    uint8_t sfu8 = 0;
    if (dim2 < in_dim2) {
      /// TODO: need opt load
      if constexpr (Swizzle) {
#pragma unroll
        for (int i = 0; i < TILE_THREAD_QUANTIZE_DIM; ++i) {
          int group_id = threadIdx.x / THREAD_NUM_PER_GROUP;
          int tid_in_group = threadIdx.x % THREAD_NUM_PER_GROUP;
          int dim3 = blockIdx.x * TILE_BLOCK_QUANTIZE_DIM +
                     group_id * GROUP_SIZE + 2 * tid_in_group + i / 2 * 8 +
                     (i % 2);
          if (dim3 < in_dim3) {
            bf16[i] = in[dim0 * in_stride0 + dim1 * in_stride1 +
                         dim2 * in_stride2 + dim3 * in_stride3];
          } else {
            reinterpret_cast<uint16_t *>(bf16)[i] = 0;
          }
        }
      } else {
#pragma unroll
        for (int i = 0; i < TILE_THREAD_QUANTIZE_DIM; ++i) {
          if (dim3 + i < in_dim3) {
            bf16[i] = in[dim0 * in_stride0 + dim1 * in_stride1 +
                         dim2 * in_stride2 + (dim3 + i) * in_stride3];
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
      {
        __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(sf_value);
        sfu8 = static_cast<uint16_t>(tmp.__x);
        sf_value = static_cast<float>(tmp);
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

      NVFP4x8 e2m1x8 = fp32x8_vec_to_e2m1x8(f32x2);
      *reinterpret_cast<NVFP4x8 *>(out + dim0 * out_stride0 +
                                   dim1 * out_stride1 + dim2 * out_stride2 +
                                   dim3 / 2 * out_stride3) = e2m1x8;
    } // end if (dim2 < in_dim2)

    if (threadIdx.x % 2 == 0) {
      reinterpret_cast<uint8_t *>(&sf[dim0 * sf_stride0 + dim1 * sf_stride1 +
                                      dim3 / BLOCK_SIZE / PACK_SF * sf_stride2 +
                                      dim2 * sf_stride3])[threadIdx.x / 2] =
          sfu8;
    }
  } // end dim0 loop
}

/// OUT layout
/// [xx, xx, cross_dim_align, reduce_dim_align / 2] x NVFP4x2
///
/// SF layout
/// [xx, xx, reduce_dim_align / 64, cross_dim_align, 4] x E4M3
void quantize_nvfp4(torch::Tensor &out, torch::Tensor &sf,
                          torch::Tensor const &in,
                          torch::Tensor const &global_scale, uint32_t cross_dim,
                          uint32_t reduce_dim, bool swizzle) {
  static_assert(TILE_BLOCK_NON_QUANTIZE_DIM <= CROSS_DIM_ALIGN_SIZE);
  static_assert(TILE_BLOCK_QUANTIZE_DIM == REDUCE_DIM_ALIGN_SIZE);
  /// Check in
  std::vector<int64_t> in_shape;
  std::vector<int64_t> in_stride;
  CHECK_INPUT(in, at::ScalarType::BFloat16, "in must be bfloat16");
  TORCH_CHECK(in.dim() <= 4 && in.dim() >= 2, "in must be 2-4D");
  for (int i = 0; i < in.dim(); i++) {
    in_shape.push_back(in.size(i));
    in_stride.push_back(in.stride(i));
  }
  /// Check global_scale
  CHECK_INPUT(global_scale, at::ScalarType::Float,
              "global_scale must be float");
  TORCH_CHECK(global_scale.dim() == 0, "global_scale must be a scalar");

  /// Check dim
  TORCH_CHECK(cross_dim < static_cast<uint32_t>(in.dim()),
              "cross_dim must be less than in.dim()");
  TORCH_CHECK(reduce_dim < static_cast<uint32_t>(in.dim()),
              "reduce_dim must be less than in.dim()");
  int64_t cross_dim_size = in_shape[cross_dim];
  int64_t reduce_dim_size = in_shape[reduce_dim];

  /// Check out
  std::vector<int64_t> out_shape;
  CHECK_OUTPUT(out, at::ScalarType::Byte, "out must be pack to uint8 tensor");
  TORCH_CHECK(out.dim() == in_shape.size(), "out must be the same dim as in");
  for (int i = 0; i < out.dim(); ++i) {
    out_shape.push_back(out.size(i));
  }
  for (uint32_t i = 0, counter = 0; i < out_shape.size() - 2; ++i) {
    if (i != cross_dim && i != reduce_dim) {
      TORCH_CHECK(in_shape[i] == out_shape[counter++],
                  "out must be the same shape as in");
    }
  }
  TORCH_CHECK(out_shape[in_shape.size() - 2] == cross_dim_size,
              "meet invalid cross dim size");
  TORCH_CHECK(out_shape[in_shape.size() - 1] ==
                  align_up(reduce_dim_size, REDUCE_DIM_ALIGN_SIZE) / 2,
              "meet invalid reduce dim size");

  /// Check sf
  CHECK_OUTPUT(sf, at::ScalarType::Float8_e4m3fn, "sf must be float8 tensor");
  /// sf layout: [xx, xx, reduce_dim_align / 64, cross_dim_align, 4] x E4M3
  TORCH_CHECK(sf.dim() == in.dim() + 1, "meet invalid sf rank");
  for (int i = 0; i < out_shape.size() - 2; ++i) {
    TORCH_CHECK(sf.size(i) == out_shape[i], "meet invalid sf shape");
  }
  TORCH_CHECK(sf.size(in_shape.size() - 2) ==
                  ceil_div(reduce_dim_size, REDUCE_DIM_ALIGN_SIZE),
              "meet invalid sf shape");
  TORCH_CHECK(sf.size(in_shape.size() - 1) ==
                  align_up(cross_dim_size, CROSS_DIM_ALIGN_SIZE),
              "meet invalid sf shape");
  TORCH_CHECK(sf.size(in_shape.size()) == PACK_SF, "meet invalid sf shape");

  /// Expand to 4D
  for (int i = in.dim(); i < 4; ++i) {
    in_shape.insert(in_shape.begin(), 1);
    in_stride.insert(in_stride.begin(), in_shape[1] * in_stride[0]);
    out_shape.insert(out_shape.begin(), 1);
  }
  cross_dim += in_shape.size() - in.dim();
  reduce_dim += in_shape.size() - in.dim();
  TORCH_CHECK(cross_dim != reduce_dim,
              "cross_dim and reduce_dim must be different");
  std::vector<int64_t> in_stride_wo_cross_reduce;
  for (int i = 0; i < in_stride.size(); ++i) {
    if (i != cross_dim && i != reduce_dim) {
      in_stride_wo_cross_reduce.push_back(in_stride[i]);
    }
  }

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  dim3 grid(align_up(reduce_dim_size, REDUCE_DIM_ALIGN_SIZE) /
                TILE_BLOCK_QUANTIZE_DIM,
            align_up(cross_dim_size, CROSS_DIM_ALIGN_SIZE) /
                TILE_BLOCK_NON_QUANTIZE_DIM,
            out_shape[1]);
  dim3 block(TILE_BLOCK_QUANTIZE_DIM / TILE_THREAD_QUANTIZE_DIM,
             TILE_BLOCK_NON_QUANTIZE_DIM / TILE_THREAD_NON_QUANTIZE_DIM);
  if (swizzle) {
    nvfp4_quantize_sm120_kernel<true><<<grid, block, 0, stream>>>(
        reinterpret_cast<NVFP4x2 *>(out.data_ptr()),
        reinterpret_cast<uint32_t *>(sf.data_ptr()),
        reinterpret_cast<const BF16 *>(in.data_ptr()),
        reinterpret_cast<const float *>(global_scale.data_ptr()), out_shape[0],
        out_shape[1], cross_dim_size, reduce_dim_size,
        in_stride_wo_cross_reduce[0], in_stride_wo_cross_reduce[1],
        in_stride[cross_dim], in_stride[reduce_dim]);
  } else {
    nvfp4_quantize_sm120_kernel<false><<<grid, block, 0, stream>>>(
        reinterpret_cast<NVFP4x2 *>(out.data_ptr()),
        reinterpret_cast<uint32_t *>(sf.data_ptr()),
        reinterpret_cast<const BF16 *>(in.data_ptr()),
        reinterpret_cast<const float *>(global_scale.data_ptr()), out_shape[0],
        out_shape[1], cross_dim_size, reduce_dim_size,
        in_stride_wo_cross_reduce[0], in_stride_wo_cross_reduce[1],
        in_stride[cross_dim], in_stride[reduce_dim]);
  }
}

} // namespace ac4k
