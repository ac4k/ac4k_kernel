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
using BF16x2 = __nv_bfloat162;
using NVFP4x2 = uint8_t;
using NVFP4x8 = uint32_t;
using F8E4M3 = uint8_t;
using F8E4M3x4 = uint32_t;

constexpr int TILE_BLOCK_REDUCE_DIM = 64;
constexpr int TILE_BLOCK_CROSS_DIM = 16;

constexpr int TILE_THREAD_REDUCE_DIM = 4;
constexpr int TILE_THREAD_CROSS_DIM = 1;

constexpr int CROSS_DIM_ALIGN_SIZE = 16;
constexpr int REDUCE_DIM_ALIGN_SIZE = 16;

static __device__ __forceinline__ F8E4M3x4 floatx4_to_e4m3x4(float4 in) {
  F8E4M3x4 val;
  asm volatile("{\n"
               ".reg .b16 lo;\n"
               ".reg .b16 hi;\n"
               "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n"
               "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n"
               "mov.b32 %0, {lo, hi};\n"
               "}"
               : "=r"(val)
               : "f"(in.x), "f"(in.y), "f"(in.z), "f"(in.w));
  return val;
}

static __device__ __forceinline__ void clear(BF16 &in) {
  reinterpret_cast<uint16_t *>(&in)[0] = 0;
}

static __device__ __forceinline__ void clear(BF16x2 &in) {
  clear(in.x);
  clear(in.y);
}

template <bool Swizzle>
__global__ void fp8_quantize_sm120_kernel(
    F8E4M3 *out, float *sf, const BF16 *in, const float *scale_max,
    int64_t in_dim0, int64_t in_dim1, int64_t /* cross-dim-size */ in_dim2,
    int64_t /* reduce-dim-size */ in_dim3, int64_t in_stride0,
    int64_t in_stride1, int64_t in_stride2, int64_t in_stride3) {
  /// SF stride(float)
  /// [xx, xx, cross_dim_align] x F32
  int64_t sf_dim0 = in_dim0;
  int64_t sf_dim1 = in_dim1;
  int64_t sf_dim2 = align_up(in_dim2, CROSS_DIM_ALIGN_SIZE);
  int64_t sf_stride2 = 1;
  int64_t sf_stride1 = sf_stride2 * sf_dim2;
  int64_t sf_stride0 = sf_stride1 * in_dim1;

  /// OUT stride
  /// [xx, xx, cross_dim, reduce_dim_align] x F8E4M3
  int64_t out_dim0 = in_dim0;
  int64_t out_dim1 = in_dim1;
  int64_t out_dim2 = in_dim2;
  int64_t out_dim3 = align_up(in_dim3, REDUCE_DIM_ALIGN_SIZE);
  int64_t out_stride3 = 1;
  int64_t out_stride2 = out_stride3 * out_dim3;
  int64_t out_stride1 = out_stride2 * out_dim2;
  int64_t out_stride0 = out_stride1 * out_dim1;

  int tid = threadIdx.x;
  int dim0 = blockIdx.z;
  int dim1 = blockIdx.y;
  int dim2 =
      blockIdx.x * TILE_BLOCK_CROSS_DIM + threadIdx.y * TILE_THREAD_CROSS_DIM;

  /// Step 0: get abs max per reduce dimension
  /// local abs max
  BF16x2 local_abs_max;
  clear(local_abs_max);
  for (int i = tid * TILE_THREAD_REDUCE_DIM;
       i < align_up(in_dim3, TILE_BLOCK_REDUCE_DIM);
       i += TILE_BLOCK_REDUCE_DIM /* blockDim.x * TILE_THREAD_REDUCE_DIM */) {
    BF16 in_bf16[TILE_THREAD_REDUCE_DIM];
#pragma unroll
    for (int j = 0; j < TILE_THREAD_REDUCE_DIM; ++j) {
      int dim3 = i + j;
      if (dim2 < in_dim2 && dim3 < in_dim3) {
        in_bf16[j] = in[dim0 * in_stride0 + dim1 * in_stride1 +
                        dim2 * in_stride2 + dim3 * in_stride3];
      } else {
        clear(in_bf16[j]);
      }
    }

    BF16x2 *in_bf16x2 = reinterpret_cast<BF16x2 *>(in_bf16);
#pragma unroll
    for (int j = 0;
         j < TILE_THREAD_REDUCE_DIM / (sizeof(BF16x2) / sizeof(BF16)); ++j) {
      local_abs_max = __hmax2(local_abs_max, __habs2(in_bf16x2[j]));
    }
  } // end loop i

  /// abs max per reduce dimension
  /// Cross-lane max
  for (int i = 1; i < TILE_BLOCK_REDUCE_DIM / TILE_THREAD_REDUCE_DIM; i *= 2) {
    local_abs_max =
        __hmax2(__shfl_xor_sync(0xffffffff, local_abs_max, i), local_abs_max);
  }
  float abs_max = __bfloat162float(__hmax(local_abs_max.x, local_abs_max.y));

  float scale = abs_max / (*scale_max);

  if (dim2 < sf_dim2) {
    sf[dim0 * sf_stride0 + dim1 * sf_stride1 + dim2 * sf_stride2] = scale;
  }

  /// Step 1: quantize per reduce dimension
  if (dim2 < in_dim2) {
    float recp_scale = (*scale_max) / abs_max;
    for (int i = tid * TILE_THREAD_REDUCE_DIM;
         i < align_up(in_dim3, REDUCE_DIM_ALIGN_SIZE);
         i += TILE_BLOCK_REDUCE_DIM /* blockDim.x * TILE_THREAD_REDUCE_DIM */) {
      BF16 in_bf16[TILE_THREAD_REDUCE_DIM];

#pragma unroll
      for (int j = 0; j < TILE_THREAD_REDUCE_DIM; ++j) {
        int dim3 = 0;
        if constexpr (Swizzle) {
          /// Swizzle for each reduce dimension
          /// reduce dimension reshape: [-1, 2, 4, 2]
          /// reduce dimension permute: [0, 2, 1, 3]
          constexpr int GROUP_SIZE = 16;
          constexpr int THREAD_NUM_PER_GROUP =
              GROUP_SIZE / TILE_THREAD_REDUCE_DIM;
          int group_id = threadIdx.x / THREAD_NUM_PER_GROUP;
          int tid_in_group = threadIdx.x % THREAD_NUM_PER_GROUP;
          dim3 = align_down(i, TILE_BLOCK_REDUCE_DIM) + group_id * GROUP_SIZE +
                 2 * tid_in_group + j / 2 * 8 + (j % 2);
        } else {
          dim3 = i + j;
        }

        if (dim3 < in_dim3) {
          in_bf16[j] = in[dim0 * in_stride0 + dim1 * in_stride1 +
                          dim2 * in_stride2 + dim3 * in_stride3];
        } else {
          clear(in_bf16[j]);
        }
      }

      /// Convert to F32: BF16 * recp_scale
      float in_f32[TILE_THREAD_REDUCE_DIM];
#pragma unroll
      for (int j = 0; j < TILE_THREAD_REDUCE_DIM; ++j) {
        in_f32[j] = __bfloat162float(in_bf16[j]) * recp_scale;
      }

      // floatx4_to_e4m3x4
      float4 *in_f32x4 = reinterpret_cast<float4 *>(in_f32);
#pragma unroll
      for (int j = 0;
           j < TILE_THREAD_REDUCE_DIM / (sizeof(float4) / sizeof(float)); ++j) {
        F8E4M3x4 out_f8x4 = floatx4_to_e4m3x4(in_f32x4[j]);

        auto *out_f8x4_ptr = reinterpret_cast<F8E4M3x4 *>(
            out + dim0 * out_stride0 + dim1 * out_stride1 + dim2 * out_stride2 +
            (i + j * sizeof(float4) / sizeof(float)) * out_stride3);
        *out_f8x4_ptr = out_f8x4;
      }
    } // end for reduce
  } // end if (dim2 < in_dim2)
}

/// OUT layout
/// [xx, xx, cross_dim_align, reduce_dim_align] x FP8E4M3
///
/// SF layout
/// [xx, xx, cross_dim_align] x F32
void fp8_quantize_sm120(torch::Tensor &out, torch::Tensor &sf,
                        torch::Tensor const &in, torch::Tensor const &scale_max,
                        uint32_t cross_dim, uint32_t reduce_dim, bool swizzle) {
  /// Check in
  std::vector<int64_t> in_shape;
  std::vector<int64_t> in_stride;
  CHECK_INPUT(in, at::ScalarType::BFloat16, "in must be bfloat16");
  TORCH_CHECK(in.dim() <= 4 && in.dim() >= 2, "in must be 2-4D");
  for (int i = 0; i < in.dim(); i++) {
    in_shape.push_back(in.size(i));
    in_stride.push_back(in.stride(i));
  }
  /// Check scale_max
  CHECK_INPUT(scale_max, at::ScalarType::Float, "scale_max must be float");
  TORCH_CHECK(scale_max.dim() == 0, "scale_max must be a scalar");

  /// Check dim
  TORCH_CHECK(cross_dim < static_cast<uint32_t>(in.dim()),
              "cross_dim must be less than in.dim()");
  TORCH_CHECK(reduce_dim < static_cast<uint32_t>(in.dim()),
              "reduce_dim must be less than in.dim()");
  int64_t cross_dim_size = in_shape[cross_dim];
  int64_t reduce_dim_size = in_shape[reduce_dim];

  /// Check out
  std::vector<int64_t> out_shape;
  CHECK_OUTPUT(out, at::ScalarType::Float8_e4m3fn,
               "out must be pack to fp8e4m3 tensor");
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
                  align_up(reduce_dim_size, REDUCE_DIM_ALIGN_SIZE),
              "meet invalid reduce dim size");

  /// Check sf
  CHECK_OUTPUT(sf, at::ScalarType::Float, "sf must be f32 tensor");
  /// sf layout: [xx, xx, cross_dim_align] x F32
  TORCH_CHECK(sf.dim() == in.dim() - 1, "meet invalid sf rank");
  for (int i = 0; i < out_shape.size() - 2; ++i) {
    TORCH_CHECK(sf.size(i) == out_shape[i], "meet invalid sf shape");
  }
  TORCH_CHECK(sf.size(sf.dim() - 1) ==
                  align_up(cross_dim_size, CROSS_DIM_ALIGN_SIZE),
              "meet invalid sf shape");

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

  dim3 grid(align_up(out_shape[2], CROSS_DIM_ALIGN_SIZE) / TILE_BLOCK_CROSS_DIM,
            out_shape[1], out_shape[0]);
  dim3 block(TILE_BLOCK_REDUCE_DIM / TILE_THREAD_REDUCE_DIM,
             TILE_BLOCK_CROSS_DIM, TILE_THREAD_CROSS_DIM);

  if (swizzle) {
    fp8_quantize_sm120_kernel<true><<<grid, block, 0, stream>>>(
        reinterpret_cast<F8E4M3 *>(out.data_ptr()),
        reinterpret_cast<float *>(sf.data_ptr()),
        reinterpret_cast<const BF16 *>(in.data_ptr()),
        reinterpret_cast<const float *>(scale_max.data_ptr()), out_shape[0],
        out_shape[1], cross_dim_size, reduce_dim_size,
        in_stride_wo_cross_reduce[0], in_stride_wo_cross_reduce[1],
        in_stride[cross_dim], in_stride[reduce_dim]);
  } else {
    fp8_quantize_sm120_kernel<false><<<grid, block, 0, stream>>>(
        reinterpret_cast<F8E4M3 *>(out.data_ptr()),
        reinterpret_cast<float *>(sf.data_ptr()),
        reinterpret_cast<const BF16 *>(in.data_ptr()),
        reinterpret_cast<const float *>(scale_max.data_ptr()), out_shape[0],
        out_shape[1], cross_dim_size, reduce_dim_size,
        in_stride_wo_cross_reduce[0], in_stride_wo_cross_reduce[1],
        in_stride[cross_dim], in_stride[reduce_dim]);
  }
}

} // namespace ac4k
