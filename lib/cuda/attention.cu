#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <mma.h>
#include <torch/all.h>

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
constexpr int ELES_PER_NVFP4x2 = 2;

constexpr int64_t HEAD_DIM_SIZE = 128;
constexpr int64_t BLOCK_SIZE = 16;
constexpr int64_t HEAD_DIM_ALIGN_SIZE = 128;

//===----------------------------------------------------------------------===//
// Define tile size for block level, warp level and atomic level
//===----------------------------------------------------------------------===//

/// Block level tile size
constexpr int TILE_DOT_BLOCK_M = 16;
constexpr int TILE_DOT0_BLOCK_M = TILE_DOT_BLOCK_M;
constexpr int TILE_DOT0_BLOCK_N = 64;
constexpr int TILE_DOT0_BLOCK_K = 128;
constexpr int TILE_DOT1_BLOCK_M = TILE_DOT0_BLOCK_M;
constexpr int TILE_DOT1_BLOCK_N = 128;
constexpr int TILE_DOT1_BLOCK_K = TILE_DOT0_BLOCK_N;

/// Warp level tile size
constexpr int TILE_DOT0_WARP_M = TILE_DOT_BLOCK_M;
constexpr int TILE_DOT0_WARP_N = TILE_DOT0_BLOCK_N;
constexpr int TILE_DOT0_WARP_K = TILE_DOT0_BLOCK_K;
constexpr int TILE_DOT1_WARP_M = TILE_DOT0_WARP_M;
constexpr int TILE_DOT1_WARP_N = 128;
constexpr int TILE_DOT1_WARP_K = TILE_DOT1_BLOCK_K;

constexpr int WARP_NUM =
    TILE_DOT0_BLOCK_M / TILE_DOT0_WARP_M * TILE_DOT0_BLOCK_N / TILE_DOT0_WARP_N;

/// Atomic level tile size
constexpr int TILE_ATOMIC_M = 16;
constexpr int TILE_ATOMIC_N = 8;
constexpr int TILE_ATOMIC_K = 64;

// Fast reciprocal.
__forceinline__ __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

// Convert 8 float32 values into 8 e2m1 values (represented as one uint32_t).
__forceinline__ __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
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
}

template <int N0, int N1>
__forceinline__ __device__ void
convert_to_nvfp4(const DFrag_F32_16x8 (&p_f32_in)[N0][N1],
                 AFrag_NVFP4_16x64 (&p_fp4_out)[N0][8 * N1 / 64],
                 uint32_t (&sf)[N0][8 * N1 / 64]) {
  int lane_id = threadIdx.x % 32;

  float rcp_global_scale_value = static_cast<float>(448 * 6);

#pragma unroll
  for (int i = 0; i < N1; i += 8) {
    uint32_t sfu32[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int k = 0; k < 8; k += 4) {
        float max = fmax(p_f32_in[0][i + k].data[j * 2],
                         p_f32_in[0][i + k].data[j * 2 + 1]);
#pragma unroll
        for (int l = 1; l < 4; ++l) {
          max = fmax(max, p_f32_in[0][i + k + l].data[j * 2]);
          max = fmax(max, p_f32_in[0][i + k + l].data[j * 2 + 1]);
        }
        max = fmax(__shfl_xor_sync(0xffffffff, max, 1), max);

        /// sf
        float sf_value = rcp_global_scale_value * (max * 0.16666666666666666f);
        uint8_t sfu8;
        {
          __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(sf_value);
          sfu8 = static_cast<uint16_t>(tmp.__x);
          sf_value = static_cast<float>(tmp);
        }
        uint8_t sfu8_other = __shfl_xor_sync(0xffffffff, sfu8, 2);
        uint16_t sfu16;
        if (lane_id % 4 < 2) {
          sfu16 = static_cast<uint16_t>(sfu8) +
                  (static_cast<uint16_t>(sfu8_other) << 8);
        } else {
          sfu16 = (static_cast<uint16_t>(sfu8) << 8) +
                  static_cast<uint16_t>(sfu8_other);
        }
        reinterpret_cast<uint16_t *>(&sfu32[j])[k / 4] = sfu16;

        float out_scale =
            sf_value != 0
                ? rcp_global_scale_value * reciprocal_approximate_ftz(sf_value)
                : 0.0f;

        float in[8];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          in[2 * m] = p_f32_in[0][i + k + m].data[j * 2] * out_scale;
          in[2 * m + 1] = p_f32_in[0][i + k + m].data[j * 2 + 1] * out_scale;
        }

        /// Convert 8xfloat32 to 8xe2m1(uint32)
        uint32_t e2m1x8 = fp32_vec_to_e2m1(in);
        p_fp4_out[0][8 * i / 64].data[j + k / 4 * 2] = e2m1x8;
      } // end loop k
    }

    uint32_t sf0 = sfu32[0];
    uint32_t sf1 = __shfl_xor_sync(0xffffffff, sfu32[1], 1);
    uint32_t sf_final = lane_id % 4 == 0 ? sf0 : sf1;
    sf[0][8 * i / 64] = sf_final;
  }

  return;
}

__global__ void nvfp4_mha_fwd_kernel(
    BF16 *O, int64_t o_b_stride, int64_t o_n_stride, int64_t o_h_stride,
    int64_t o_d_stride, const NVFP4x2 *Q, int64_t q_b_stride,
    int64_t q_n_stride, int64_t q_h_stride, int64_t q_d_stride,
    const E4M3 *Q_SF, int64_t q_sf_quantize_stride,
    int64_t q_sf_non_quantize_stride, const NVFP4x2 *K, int64_t k_b_stride,
    int64_t k_n_stride, int64_t k_h_stride, int64_t k_d_stride,
    const E4M3 *K_SF, int64_t k_sf_quantize_stride,
    int64_t k_sf_non_quantize_stride, const NVFP4x2 *V, int64_t v_b_stride,
    int64_t v_h_stride, int64_t v_d_stride, int64_t v_n_stride,
    const E4M3 *V_SF, int64_t v_sf_quantize_stride,
    int64_t v_sf_non_quantize_stride, const float *alpha0, const float *alpha1,
    int64_t B, int64_t H, int64_t Nq, int64_t Nkv, int64_t Dqk, int64_t Dv,
    float qk_norm) {
  int block_b = blockIdx.x;                    // batch
  int block_h = blockIdx.y;                    // head
  int block_n = blockIdx.z * TILE_DOT_BLOCK_M; // seq

  int lane_id = threadIdx.x % 32;
  int warp_id = threadIdx.y;

  // dot0 = q @ k
  // dot1 = p @ v
  int DOT0_M = Nq;
  int DOT0_N = Nkv;
  int DOT0_K = Dqk;
  int DOT1_M = Nq;
  int DOT1_N = Dv;
  int DOT1_K = Nkv;
  int block_dot_m = block_n;
  int block_dot_n = 0;
  int block_dot0_m = block_dot_m;
  int block_dot0_n = block_dot_n;
  int block_dot1_m = block_dot_m;
  int block_dot1_n = block_dot_n;
  int warp_dot0_m = 0;
  int warp_dot0_n = 0;
  int warp_dot1_m = 0;
  int warp_dot1_n = 0;

  //===--------------------------------------------------------------------===//
  // Define fragment
  //===--------------------------------------------------------------------===//

  /// Q
  AFrag_NVFP4_16x64 q_frag[TILE_DOT0_WARP_M / TILE_ATOMIC_M]
                          [TILE_DOT0_WARP_K / TILE_ATOMIC_K];
  uint32_t q_sf_frag[TILE_DOT0_WARP_M / TILE_ATOMIC_M]
                    [TILE_DOT0_WARP_K / TILE_ATOMIC_K];
  /// max(row max of S=Q @ K)
  float max0 = -INFINITY;
  float max1 = -INFINITY;
  /// l（sum of exp)
  float l0 = 0;
  float l1 = 0;
  /// o: P @ V
  DFrag_F32_16x8 o_frag[TILE_DOT1_WARP_M / TILE_ATOMIC_M]
                       [TILE_DOT1_WARP_N / TILE_ATOMIC_N];
#pragma unroll
  for (int i = 0; i < TILE_DOT1_WARP_M / TILE_ATOMIC_M; ++i) {
#pragma unroll
    for (int j = 0; j < TILE_DOT1_WARP_N / TILE_ATOMIC_N; ++j) {
#pragma unroll
      for (int k = 0; k < o_frag[i][j].REGISTERS_PER_THREAD; ++k) {
        /// Clear o fragment
        o_frag[i][j].data[k] = 0.0f;
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Pre-load Q && Q_SF
  //===--------------------------------------------------------------------===//

#pragma unroll
  for (int i = 0; i < TILE_DOT0_WARP_M / TILE_ATOMIC_M; ++i) {
#pragma unroll
    for (int j = 0; j < TILE_DOT0_WARP_K / TILE_ATOMIC_K; ++j) {
#pragma unroll
      for (int idx = 0; idx < q_frag[i][j].REGISTERS_PER_THREAD; ++idx) {
        // TODO(check out-of-bound)
        // m map to N(seq)
        int m = block_dot0_m + warp_dot0_m + i * TILE_ATOMIC_M +
                q_frag[i][j].get_row_with_reg(lane_id, idx);
        // k map to Dqk(emb)
        int k = j * TILE_ATOMIC_K + q_frag[i][j].get_col_with_reg(lane_id, idx);
        const auto *q_tile = Q + block_b * q_b_stride + block_h * q_h_stride +
                             m * q_n_stride + k / ELES_PER_NVFP4x2 * q_d_stride;
        q_frag[i][j].data[idx] =
            *reinterpret_cast<const AFrag_NVFP4_16x64::REG_TYPE *>(q_tile);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < TILE_DOT0_WARP_M / TILE_ATOMIC_M; ++i) {
#pragma unroll
    for (int j = 0; j < TILE_DOT0_WARP_K / TILE_ATOMIC_K; ++j) {
      if (lane_id % 4 < 2) {
        // m map to non-quantize
        int m = block_dot0_m + warp_dot0_m + i * TILE_ATOMIC_M +
                (lane_id % 2) * 8 + lane_id / 4;
        // k map to quantize
        int k = j * TILE_ATOMIC_K;
        const auto *q_sf_tile =
            Q_SF + (k / (BLOCK_SIZE * 4)) * q_sf_quantize_stride +
            (block_b * Nq * H + m * H + block_h) * q_sf_non_quantize_stride;
        q_sf_frag[i][j] = *reinterpret_cast<const uint32_t *>(q_sf_tile);
      }
    }
  }

  for (int dot0_n = 0; dot0_n < DOT0_N; dot0_n += TILE_DOT0_BLOCK_N) {
    block_dot0_n = dot0_n;

    /// S
    DFrag_F32_16x8 s_frag[TILE_DOT0_WARP_M / TILE_ATOMIC_M]
                         [TILE_DOT0_WARP_N / TILE_ATOMIC_N];
#pragma unroll
    for (int i = 0; i < TILE_DOT0_WARP_M / TILE_ATOMIC_M; ++i) {
#pragma unroll
      for (int j = 0; j < TILE_DOT0_WARP_N / TILE_ATOMIC_N; ++j) {
#pragma unroll
        for (int k = 0; k < s_frag[i][j].REGISTERS_PER_THREAD; ++k) {
          /// Clear s fragment
          s_frag[i][j].data[k] = 0.0f;
        }
      }
    }

    /// Step 0: S = Q @ K
    ///      16x64x128=> 16*8*2=256 cycles

#pragma unroll
    for (int dot0_k = 0; dot0_k < TILE_DOT0_BLOCK_K; dot0_k += TILE_ATOMIC_K) {
#pragma unroll
      for (int dot0_atomic_n = 0; dot0_atomic_n < TILE_DOT0_WARP_N;
           dot0_atomic_n += TILE_ATOMIC_N) {
        /// Load K
        BFrag_NVFP4_64x8 k_frag;
#pragma unroll
        for (int idx = 0; idx < k_frag.REGISTERS_PER_THREAD; ++idx) {
          // map k to K's Dk(emb)
          int k = dot0_k + k_frag.get_row_with_reg(lane_id, idx);
          // map n to K's N(seq)
          int n = block_dot0_n + warp_dot0_n + dot0_atomic_n +
                  k_frag.get_col_with_reg(lane_id, idx);
          const auto *k_tile = K + block_b * k_b_stride + block_h * k_h_stride +
                               n * k_n_stride +
                               k / ELES_PER_NVFP4x2 * k_d_stride;
          k_frag.data[idx] =
              *reinterpret_cast<const BFrag_NVFP4_64x8::REG_TYPE *>(k_tile);
        }

        /// Load K_SF
        uint32_t k_sf_frag = 0;
        if (lane_id % 4 == 0) {
          // map k to quantize
          int k = dot0_k;
          // map n to non-quantize
          int n = block_dot0_n + warp_dot0_n + dot0_atomic_n + (lane_id / 4);
          const auto *k_sf_tile =
              K_SF + (k / (BLOCK_SIZE * 4)) * k_sf_quantize_stride +
              (block_b * Nkv * H + n * H + block_h) * k_sf_non_quantize_stride;
          k_sf_frag = *reinterpret_cast<const uint32_t *>(k_sf_tile);
        }

        /// Dot0
        fma(s_frag[0][dot0_atomic_n / TILE_ATOMIC_N].data[0],
            s_frag[0][dot0_atomic_n / TILE_ATOMIC_N].data[1],
            s_frag[0][dot0_atomic_n / TILE_ATOMIC_N].data[2],
            s_frag[0][dot0_atomic_n / TILE_ATOMIC_N].data[3],
            q_frag[0][dot0_k / TILE_ATOMIC_K].data[0],
            q_frag[0][dot0_k / TILE_ATOMIC_K].data[1],
            q_frag[0][dot0_k / TILE_ATOMIC_K].data[2],
            q_frag[0][dot0_k / TILE_ATOMIC_K].data[3], k_frag.data[0],
            k_frag.data[1], s_frag[0][dot0_atomic_n / TILE_ATOMIC_N].data[0],
            s_frag[0][dot0_atomic_n / TILE_ATOMIC_N].data[1],
            s_frag[0][dot0_atomic_n / TILE_ATOMIC_N].data[2],
            s_frag[0][dot0_atomic_n / TILE_ATOMIC_N].data[3],
            q_sf_frag[0][dot0_k / TILE_ATOMIC_K], k_sf_frag);
      } // end loop dot0_atomic_n
    } // end loop dot0_k

    /// Step 1: S = S * alpha0(nvfp4 global scale) * qk_norm(rsqrt(dk))
    ///         8 * 4 = 32 cycles
    float qk_scale = *alpha0 * qk_norm;
#pragma unroll
    for (int i = 0; i < TILE_DOT0_WARP_M / TILE_ATOMIC_M; ++i) {
#pragma unroll
      for (int j = 0; j < TILE_DOT0_WARP_N / TILE_ATOMIC_N; ++j) {
#pragma unroll
        for (int k = 0; k < s_frag[i][j].REGISTERS_PER_THREAD; ++k) {
          s_frag[i][j].data[k] *= qk_scale;
        }
      }
    }

    /// Step 2: max = rowmax(S): 40 cycles
    float max_new0 = max0;
    float max_new1 = max1;
#pragma unroll
    for (int i = 0; i < TILE_DOT0_WARP_N / TILE_ATOMIC_N; ++i) {
      max_new0 = fmaxf(max_new0, s_frag[0][i].data[0]);
      max_new0 = fmaxf(max_new0, s_frag[0][i].data[1]);
      max_new1 = fmaxf(max_new1, s_frag[0][i].data[2]);
      max_new1 = fmaxf(max_new1, s_frag[0][i].data[3]);
    }
    max_new0 = fmaxf(__shfl_xor_sync(0xffffffff, max_new0, 1), max_new0);
    max_new0 = fmaxf(__shfl_xor_sync(0xffffffff, max_new0, 2), max_new0);
    max_new1 = fmaxf(__shfl_xor_sync(0xffffffff, max_new1, 1), max_new1);
    max_new1 = fmaxf(__shfl_xor_sync(0xffffffff, max_new1, 2), max_new1);

    /// Step 3: p = exp(S - max)
    ///         160 cycles
    DFrag_F32_16x8 p_frag[TILE_DOT0_WARP_M / TILE_ATOMIC_M]
                         [TILE_DOT0_WARP_N / TILE_ATOMIC_N];
#pragma unroll
    for (int i = 0; i < TILE_DOT0_WARP_N / TILE_ATOMIC_N; ++i) {
      p_frag[0][i].data[0] = expf(s_frag[0][i].data[0] - max_new0);
      p_frag[0][i].data[1] = expf(s_frag[0][i].data[1] - max_new0);
      p_frag[0][i].data[2] = expf(s_frag[0][i].data[2] - max_new1);
      p_frag[0][i].data[3] = expf(s_frag[0][i].data[3] - max_new1);
    }

    /// Step 4: l = sum(p)
    ///         14 + 4 * 2 + 2 * 7 = 36 cycles
    float l_new0 = p_frag[0][0].data[0] + p_frag[0][0].data[1];
    float l_new1 = p_frag[0][0].data[2] + p_frag[0][0].data[3];
#pragma unroll
    for (int i = 1; i < TILE_DOT0_WARP_N / TILE_ATOMIC_N; ++i) {
      l_new0 += p_frag[0][i].data[0];
      l_new0 += p_frag[0][i].data[1];
      l_new1 += p_frag[0][i].data[2];
      l_new1 += p_frag[0][i].data[3];
    }
    l_new0 += __shfl_xor_sync(0xffffffff, l_new0, 1);
    l_new0 += __shfl_xor_sync(0xffffffff, l_new0, 2);
    l_new1 += __shfl_xor_sync(0xffffffff, l_new1, 1);
    l_new1 += __shfl_xor_sync(0xffffffff, l_new1, 2);
    // Update l
    l0 = expf(max0 - max_new0) * l0 + l_new0;
    l1 = expf(max1 - max_new1) * l1 + l_new1;

    /// Step 4: o = expf(max - max_new) * o
    ///         4 * 8 + 2 = 34

#pragma unroll
    for (int i = 0; i < TILE_DOT1_WARP_M / TILE_ATOMIC_M; ++i) {
#pragma unroll
      for (int j = 0; j < TILE_DOT1_WARP_N / TILE_ATOMIC_N; ++j) {
        o_frag[i][j].data[0] *= expf(max0 - max_new0);
        o_frag[i][j].data[1] *= expf(max0 - max_new0);
        o_frag[i][j].data[2] *= expf(max1 - max_new1);
        o_frag[i][j].data[3] *= expf(max1 - max_new1);
      }
    }
    /// Update max
    max0 = max_new0;
    max1 = max_new1;

    /// Step 5: quantize P from f32 to nvfp4
    AFrag_NVFP4_16x64 p_fp4_frag[TILE_DOT1_WARP_M / TILE_ATOMIC_M]
                                [TILE_DOT1_WARP_K / TILE_ATOMIC_K];
    uint32_t p_fp4_sf_frag[TILE_DOT1_WARP_M / TILE_ATOMIC_M]
                          [TILE_DOT1_WARP_K / TILE_ATOMIC_K];
    convert_to_nvfp4(p_frag, p_fp4_frag, p_fp4_sf_frag);

    /// Step 6: o = P @ V
    ///         256 cycles
    int dot1_k = dot0_n;

#pragma unroll
    for (int dot1_atomic_k = 0; dot1_atomic_k < TILE_DOT1_BLOCK_K;
         dot1_atomic_k += TILE_ATOMIC_K) {
#pragma unroll
      for (int dot1_atomic_n = 0; dot1_atomic_n < TILE_DOT1_WARP_N;
           dot1_atomic_n += TILE_ATOMIC_N) {
        /// Load V
        BFrag_NVFP4_64x8 v_frag;
#pragma unroll
        for (int idx = 0; idx < v_frag.REGISTERS_PER_THREAD; ++idx) {
          // map k to V's N(seq)
          int k =
              dot1_k + dot1_atomic_k + v_frag.get_row_with_reg(lane_id, idx);
          // map n to V's D(emb)
          int n = block_dot1_n + warp_dot1_n + dot1_atomic_n +
                  v_frag.get_col_with_reg(lane_id, idx);

          const auto *v_tile = V + block_b * v_b_stride + block_h * v_h_stride +
                               n * v_d_stride +
                               k / ELES_PER_NVFP4x2 * v_n_stride;
          v_frag.data[idx] =
              *reinterpret_cast<const BFrag_NVFP4_64x8::REG_TYPE *>(v_tile);
        }

        /// Load V_SF
        /// V_SF layout: [N/64, B * H * D, 4]xuint8
        uint32_t v_sf_frag = 0;
        if (lane_id % 4 == 0) {
          // map n to V's D(emb)
          int n = block_dot1_n + warp_dot1_n + dot1_atomic_n + (lane_id / 4);
          // map k to V's N(seq)
          int k = dot1_k + dot1_atomic_k;

          const auto *v_sf_tile =
              V_SF + (k / (BLOCK_SIZE * 4)) * v_sf_quantize_stride +
              (block_b * H * Dv + block_h * Dv + n) * v_sf_non_quantize_stride;
          v_sf_frag = *reinterpret_cast<const uint32_t *>(v_sf_tile);
        }

        /// Dot1
        fma(o_frag[0][dot1_atomic_n / TILE_ATOMIC_N].data[0],
            o_frag[0][dot1_atomic_n / TILE_ATOMIC_N].data[1],
            o_frag[0][dot1_atomic_n / TILE_ATOMIC_N].data[2],
            o_frag[0][dot1_atomic_n / TILE_ATOMIC_N].data[3],
            p_fp4_frag[0][dot1_atomic_k / TILE_ATOMIC_K].data[0],
            p_fp4_frag[0][dot1_atomic_k / TILE_ATOMIC_K].data[1],
            p_fp4_frag[0][dot1_atomic_k / TILE_ATOMIC_K].data[2],
            p_fp4_frag[0][dot1_atomic_k / TILE_ATOMIC_K].data[3],
            v_frag.data[0], v_frag.data[1],
            o_frag[0][dot1_atomic_n / TILE_ATOMIC_N].data[0],
            o_frag[0][dot1_atomic_n / TILE_ATOMIC_N].data[1],
            o_frag[0][dot1_atomic_n / TILE_ATOMIC_N].data[2],
            o_frag[0][dot1_atomic_n / TILE_ATOMIC_N].data[3],
            p_fp4_sf_frag[0][dot1_atomic_k / TILE_ATOMIC_K], v_sf_frag);
      } // end loop dot1_atomic_n
    } // end loop dot1_atomic_k
  } // end loop dot0_n/dot1_k

  //===------------------------------------------------------------------===//
  // Epilogue
  //===------------------------------------------------------------------===//

  /// step 8: o = o / l * alpha1(V global scale) * p_scale(1/(448 * 6))

  for (int i = 0; i < TILE_DOT1_WARP_M / TILE_ATOMIC_M; ++i) {
    for (int j = 0; j < TILE_DOT1_WARP_N / TILE_ATOMIC_N; ++j) {
      float scale0 = (1.0f / l0) * (1.0f / (448 * 6)) * *alpha1;
      float scale1 = (1.0f / l1) * (1.0f / (448 * 6)) * *alpha1;
      o_frag[i][j].data[0] *= scale0;
      o_frag[i][j].data[1] *= scale0;
      o_frag[i][j].data[2] *= scale1;
      o_frag[i][j].data[3] *= scale1;
    }
  }

  /// Step 9: Write back to O

#pragma unroll
  for (int i = 0; i < TILE_DOT1_WARP_M / TILE_ATOMIC_M; ++i) {
#pragma unroll
    for (int j = 0; j < TILE_DOT1_WARP_N / TILE_ATOMIC_N; ++j) {
#pragma unroll
      for (int idx = 0; idx < o_frag[i][j].REGISTERS_PER_THREAD; ++idx) {
        int thread_m =
            i * TILE_ATOMIC_M + o_frag[i][j].get_row_with_reg(lane_id, idx);
        // map m to O's N(seq)
        int m = block_dot_m + warp_dot1_m + thread_m;
        int thread_n =
            j * TILE_ATOMIC_N + o_frag[i][j].get_col_with_reg(lane_id, idx);
        // map n to O's D(emb)
        int n = block_dot_n + warp_dot1_n + thread_n;
        auto *o_tile = O + block_b * o_b_stride + m * o_n_stride +
                       block_h * o_h_stride + n * o_d_stride;
        *o_tile = __float2bfloat16_rn(o_frag[i][j].data[idx]);
      }
    }
  }
}

//===--------------------------------------------------------------------===//
// ac4k mha attention with nvfp4 acceleration
// Q: [B, Nq,  H, Dqk/2]xNVFP4
// K: [B, Nkv, H, Dqk/2]xNVFP4
// V: [B, H, Dv, Nkv/2]xNVFP4
// O: [B, Nq,  H, Dv/2]xNVFP4
// Q_SF: [Dqk/64, B*Nq*H, 4]xUINT8
// K_SF: [Dqk/64, B*Nkv*H, 4]xUINT8
// V_SF: [Nkv/64, B*H*Dv, 4]xUINT8
// alpha0: Q @ K global scale
// alpha1: V global scale
//===--------------------------------------------------------------------===//

void nvfp4_mha_fwd(torch::Tensor &o, torch::Tensor &q, torch::Tensor &q_sf,
                   torch::Tensor &k, torch::Tensor &k_sf, torch::Tensor &v,
                   torch::Tensor &v_sf, torch::Tensor &alpha0,
                   torch::Tensor &alpha1) {
  /// CHECK Q & Q_SF
  CHECK_INPUT(q, at::ScalarType::Byte, "Q must be pack to uint8 tensor");
  TORCH_CHECK(q.dim() == 4, "Q must be a 4D tensor");
  int64_t B = q.size(0);
  int64_t Nq = q.size(1);
  int64_t H = q.size(2);
  int64_t Dqk = q.size(3) * ELES_PER_NVFP4x2;
  /// TODO(need remove limit)
  TORCH_CHECK(Dqk == HEAD_DIM_ALIGN_SIZE, "Dqk must be 128");
  CHECK_INPUT(q_sf, at::ScalarType::Float8_e4m3fn,
              "q_sf must be a f8e4m3 tensor");
  TORCH_CHECK(q_sf.dim() == 3, "Q_SF must be a 3D tensor");
  TORCH_CHECK(q_sf.size(0) == ceil_div(Dqk, BLOCK_SIZE * 4),
              "Meet invalid q_sf size(0) with ", q_sf.size(0),
              "==", ceil_div(Dqk, BLOCK_SIZE * 4));
  TORCH_CHECK(q_sf.size(1) == B * Nq * H, "Meet invalid q_sf size(1) with ",
              q_sf.size(1), "==", B * Nq * H);
  TORCH_CHECK(q_sf.size(2) == 4, "Meet invalid q_sf size(2) with ",
              q_sf.size(2), "==", 4);

  /// CHECK K & K_SF
  CHECK_INPUT(k, at::ScalarType::Byte, "K must be pack to uint8 tensor");
  TORCH_CHECK(k.dim() == 4, "K must be a 4D tensor");
  TORCH_CHECK(k.size(0) == B, "K must have the same batch size as Q");
  int64_t Nkv = k.size(1);
  TORCH_CHECK(k.size(2) == H, "K must have the same head number as Q");
  TORCH_CHECK(k.size(3) * ELES_PER_NVFP4x2 == Dqk,
              "K must have the same dim as Q");
  CHECK_INPUT(k_sf, at::ScalarType::Float8_e4m3fn,
              "K_SF must be a f8e4m3 tensor");
  TORCH_CHECK(k_sf.dim() == 3, "k_sf must be a 3D tensor");
  TORCH_CHECK(k_sf.size(0) == ceil_div(Dqk, BLOCK_SIZE * 4),
              "Meet invalid k_sf size(0) with ", k_sf.size(0),
              "==", ceil_div(Dqk, BLOCK_SIZE * 4));
  TORCH_CHECK(k_sf.size(1) == B * Nkv * H, "Meet invalid k_sf size(1) with ",
              k_sf.size(1), "==", B * Nkv * H);
  TORCH_CHECK(k_sf.size(2) == 4, "Meet invalid k_sf size(2) with ",
              k_sf.size(2), "==", 4);

  /// CHECK V & V_SF
  CHECK_INPUT(v, at::ScalarType::Byte, "V must be pack to uint8 tensor");
  TORCH_CHECK(v.dim() == 4, "V must be a 4D tensor");
  TORCH_CHECK(v.size(0) == B, "V must have the same batch size as Q");
  TORCH_CHECK(v.size(1) == H, "V must have the same head number as Q");
  int64_t Dv = v.size(2);
  TORCH_CHECK(v.size(3) * ELES_PER_NVFP4x2 == align_up(Nkv, BLOCK_SIZE * 4),
              "V must have the same sequence length as K");
  /// TODO(need remove limit)
  TORCH_CHECK(Dv == HEAD_DIM_ALIGN_SIZE, "Dqk must be 128");
  CHECK_INPUT(v_sf, at::ScalarType::Float8_e4m3fn,
              "V_SF must be a f8e4m3 tensor");
  TORCH_CHECK(v_sf.dim() == 3, "V_SF must be a 3D tensor");
  TORCH_CHECK(v_sf.size(0) == ceil_div(Nkv, BLOCK_SIZE * 4),
              "Meet invalid v_sf size(0) with ", v_sf.size(0),
              "==", ceil_div(Nkv, BLOCK_SIZE * 4));
  TORCH_CHECK(v_sf.size(1) == B * H * Dv, "Meet invalid v_sf size(1) with ",
              v_sf.size(1), "==", B * H * Dv);
  TORCH_CHECK(v_sf.size(2) == 4, "Meet invalid v_sf size(2) with ",
              v_sf.size(2), "==", 4);

  /// CHECK O
  CHECK_INPUT(o, at::ScalarType::BFloat16, "O must be a bfloat16 tensor");
  TORCH_CHECK(o.dim() == 4, "O must be a 4D tensor");
  TORCH_CHECK(o.size(0) == B, "O must have the same batch size as Q");
  TORCH_CHECK(o.size(1) == Nq, "O must have the same sequence length as Q");
  TORCH_CHECK(o.size(2) == H, "O must have the same head number as Q");
  TORCH_CHECK(o.size(3) == Dv, "O must have the same dim as V");

  /// CHECK alpha0 & alpha1
  CHECK_INPUT(alpha0, at::ScalarType::Float, "alpha0 must be a float tensor");
  TORCH_CHECK(alpha0.dim() == 0, "alpha0 must be a scalar");
  CHECK_INPUT(alpha1, at::ScalarType::Float, "alpha1 must be a float tensor");
  TORCH_CHECK(alpha1.dim() == 0, "alpha1 must be a scalar");

  /// Get CUDA stream
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  dim3 grid(B, H, ceil_div(Nq, static_cast<int64_t>(TILE_DOT_BLOCK_M)));
  dim3 block(32, WARP_NUM);
  nvfp4_mha_fwd_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<BF16 *>(o.data_ptr()), o.stride(0), o.stride(1),
      o.stride(2), o.stride(3), reinterpret_cast<const NVFP4x2 *>(q.data_ptr()),
      q.stride(0), q.stride(1), q.stride(2), q.stride(3),
      reinterpret_cast<const E4M3 *>(q_sf.data_ptr()), q_sf.stride(0),
      q_sf.stride(1), reinterpret_cast<const NVFP4x2 *>(k.data_ptr()),
      k.stride(0), k.stride(1), k.stride(2), k.stride(3),
      reinterpret_cast<const E4M3 *>(k_sf.data_ptr()), k_sf.stride(0),
      k_sf.stride(1), reinterpret_cast<const NVFP4x2 *>(v.data_ptr()),
      v.stride(0), v.stride(1), v.stride(2), v.stride(3),
      reinterpret_cast<const E4M3 *>(v_sf.data_ptr()), v_sf.stride(0),
      v_sf.stride(1), reinterpret_cast<const float *>(alpha0.data_ptr()),
      reinterpret_cast<const float *>(alpha1.data_ptr()), B, H, Nq, Nkv, Dqk,
      Dv, 1.0f / std::sqrt(static_cast<float>(Dqk)));
}

} // namespace ac4k
