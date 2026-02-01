#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cmath>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
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
#define CHECK_OUTPUT(x, st, m)                                                 \
  CHECK_TH_CUDA(x, m);                                                         \
  CHECK_TYPE(x, st, m)
#define CHECK_SCALAR(x, st, m)                                                 \
  CHECK_TH_CUDA(x, m);                                                         \
  CHECK_TYPE(x, st, m);                                                        \
  TORCH_CHECK(x.dim() == 0, m)

namespace ac4k {

using BF16 = __nv_bfloat16;
using F8E4M3 = uint8_t;
using F8E4M3x4 = uint32_t;
using INT8 = int8_t;

constexpr int64_t MAX_HEAD_DIM_SIZE = 128;

template <int HEAD_DIM_QK, int HEAD_DIM_V, int Q_QUANTIZE_BLOCK_SIZE_,
          int K_QUANTIZE_BLOCK_SIZE_>
struct Policy {
  using Q_TYPE = INT8;
  using K_TYPE = INT8;
  using V_TYPE = F8E4M3;
  using Q_SF_TYPE = float;
  using K_SF_TYPE = float;
  using V_SF_TYPE = float;

  //===--------------------------------------------------------------------===//
  // Quantize
  //===--------------------------------------------------------------------===//

  static constexpr int Q_QUANTIZE_BLOCK_SIZE = Q_QUANTIZE_BLOCK_SIZE_;
  static constexpr int K_QUANTIZE_BLOCK_SIZE = K_QUANTIZE_BLOCK_SIZE_;

  //===--------------------------------------------------------------------===//
  // Define tile size for block level, warp level and atomic level
  //===--------------------------------------------------------------------===//

  /// Block level tile size
  static constexpr int TILE_DOT_BLOCK_M = 128;
  static constexpr int TILE_DOT0_BLOCK_M = TILE_DOT_BLOCK_M;
  static constexpr int TILE_DOT0_BLOCK_N = 128;
  static constexpr int TILE_DOT0_BLOCK_K = HEAD_DIM_QK;
  static constexpr int TILE_DOT1_BLOCK_M = TILE_DOT0_BLOCK_M;
  static constexpr int TILE_DOT1_BLOCK_N = HEAD_DIM_V;
  static constexpr int TILE_DOT1_BLOCK_K = TILE_DOT0_BLOCK_N;

  /// Warp level tile size
  static constexpr int TILE_DOT0_WARP_M = 16;
  static constexpr int TILE_DOT0_WARP_N = TILE_DOT0_BLOCK_N;
  static constexpr int TILE_DOT0_WARP_K = TILE_DOT0_BLOCK_K;
  static constexpr int TILE_DOT1_WARP_M = TILE_DOT0_WARP_M;
  static constexpr int TILE_DOT1_WARP_N = TILE_DOT1_BLOCK_N;
  static constexpr int TILE_DOT1_WARP_K = TILE_DOT1_BLOCK_K;

  static constexpr int CONSUMER_WARP_NUM = TILE_DOT0_BLOCK_M /
                                           TILE_DOT0_WARP_M *
                                           TILE_DOT0_BLOCK_N / TILE_DOT0_WARP_N;
  static constexpr int PRODUCER_WARP_NUM = 4;
  static constexpr int WARP_SIZE = 32;

  /// Atomic levle tile size
  /// Dot0: Q @ V i8.i8.i32
  static constexpr int TILE_DOT0_ATOMIC_M = 16;
  static constexpr int TILE_DOT0_ATOMIC_N = 8;
  static constexpr int TILE_DOT0_ATOMIC_K = 32;
  /// Dot1: Q @ K e4m3.e4m3.f16
  static constexpr int TILE_DOT1_ATOMIC_M = 16;
  static constexpr int TILE_DOT1_ATOMIC_N = 8;
  static constexpr int TILE_DOT1_ATOMIC_K = 32;

  /// Tile size
  static constexpr int TILE_Q_BLOCK_ELES =
      TILE_DOT0_BLOCK_M * TILE_DOT0_BLOCK_K;
  static constexpr int TILE_K_BLOCK_ELES =
      TILE_DOT0_BLOCK_K * TILE_DOT0_BLOCK_N;
  static constexpr int TILE_V_BLOCK_ELES =
      TILE_DOT1_BLOCK_K * TILE_DOT1_BLOCK_N;
  static constexpr int TILE_Q_SF_BLOCK_ELES = TILE_DOT0_BLOCK_M;
  static constexpr int TILE_K_SF_BLOCK_ELES = TILE_DOT0_BLOCK_N;
  static constexpr int TILE_V_SF_BLOCK_ELES = TILE_DOT1_BLOCK_N;
  static constexpr int TILE_Q_BLOCK_SIZE = TILE_Q_BLOCK_ELES * sizeof(Q_TYPE);
  static constexpr int TILE_K_BLOCK_SIZE = TILE_K_BLOCK_ELES * sizeof(K_TYPE);
  static constexpr int TILE_V_BLOCK_SIZE = TILE_V_BLOCK_ELES * sizeof(V_TYPE);
  static constexpr int TILE_Q_SF_BLOCK_SIZE =
      TILE_Q_SF_BLOCK_ELES * sizeof(Q_SF_TYPE);
  static constexpr int TILE_K_SF_BLOCK_SIZE =
      TILE_K_SF_BLOCK_ELES * sizeof(K_SF_TYPE);
  static constexpr int TILE_V_SF_BLOCK_SIZE =
      TILE_V_SF_BLOCK_ELES * sizeof(V_SF_TYPE);

  /// P & V scale factor
  static constexpr float V_SCALE_MAX = 2.25f;
  /// EXP_OFFSET < log2(FP16 / TILE_DOT0_BLOCK_N / V_SCALE_MAX)
  static constexpr float EXP_OFFSET = 7.5f;

  //===--------------------------------------------------------------------===//
  // Define mutiple buffer stage
  //===--------------------------------------------------------------------===//

  static constexpr int STAGE = 2;

  //===--------------------------------------------------------------------===//
  // Consumer & Producer
  //===--------------------------------------------------------------------===//

  static constexpr int CONSUMER_THREAD_NUM = CONSUMER_WARP_NUM * WARP_SIZE;
  static constexpr int PRODUCER_THREAD_NUM = PRODUCER_WARP_NUM * WARP_SIZE;
  static constexpr int THREAD_NUM = CONSUMER_THREAD_NUM + PRODUCER_THREAD_NUM;

  //===--------------------------------------------------------------------===//
  // Swizzle
  //===--------------------------------------------------------------------===//

  static constexpr CUtensorMapSwizzle SWIZZLE_Q =
      get_swizzle<TILE_DOT0_BLOCK_K, sizeof(Q_TYPE)>();
  static constexpr CUtensorMapSwizzle SWIZZLE_K =
      get_swizzle<TILE_DOT0_BLOCK_K, sizeof(K_TYPE)>();
  static constexpr CUtensorMapSwizzle SWIZZLE_V =
      get_swizzle<TILE_DOT1_BLOCK_K, sizeof(V_TYPE)>();

  //===--------------------------------------------------------------------===//
  // Shared memory layout
  //===--------------------------------------------------------------------===//
  // Q/K0/K1/V0/V1/Q_SF/K_SF0/K_SF1/V_SF
  static constexpr int Q_SMEM_OFF = 0;
  static constexpr int K_SMEM_OFF = Q_SMEM_OFF + TILE_Q_BLOCK_SIZE;
  static constexpr int V_SMEM_OFF = K_SMEM_OFF + TILE_K_BLOCK_SIZE * STAGE;
  static constexpr int V_SF_SMEM_OFF = V_SMEM_OFF + TILE_V_BLOCK_SIZE * STAGE;
  static constexpr int SMEM_SIZE =
      TILE_Q_BLOCK_SIZE + TILE_K_BLOCK_SIZE * STAGE +
      TILE_V_BLOCK_SIZE * STAGE + TILE_V_SF_BLOCK_SIZE;
  static __forceinline__ __device__ Q_TYPE *get_q_stage_mem(char *smem,
                                                            int stage) {
    (void)stage;
    return reinterpret_cast<Q_TYPE *>(smem + Q_SMEM_OFF);
  }
  static __forceinline__ __device__ K_TYPE *get_k_stage_mem(char *smem,
                                                            int stage) {
    return reinterpret_cast<K_TYPE *>(smem + K_SMEM_OFF +
                                      TILE_K_BLOCK_SIZE * stage);
  }
  static __forceinline__ __device__ V_TYPE *get_v_stage_mem(char *smem,
                                                            int stage) {
    return reinterpret_cast<V_TYPE *>(smem + V_SMEM_OFF +
                                      TILE_V_BLOCK_SIZE * stage);
  }
  static __forceinline__ __device__ V_SF_TYPE *get_v_sf_stage_mem(char *smem,
                                                                  int stage) {
    (void)stage;
    return reinterpret_cast<V_SF_TYPE *>(smem + V_SF_SMEM_OFF);
  }
};

//===----------------------------------------------------------------------===//
// Create tma tensor map
//===----------------------------------------------------------------------===//

template <int Dim0, int Dim1, int Dim2, int Dim3, CUtensorMapDataType DataType,
          typename T, CUtensorMapSwizzle Swizzle = CU_TENSOR_MAP_SWIZZLE_NONE>
static __forceinline__ CUtensorMap create_4d_tensor_map(const T *gmem_ptr,
                                                        uint64_t global_dim0,
                                                        uint64_t global_dim1,
                                                        uint64_t global_dim2,
                                                        uint64_t global_dim3) {
  int BPE = sizeof(T);

  CUtensorMap template_tensor_map{};
  void *gmem_address = (void *)gmem_ptr;
  uint64_t gmem_prob_shape[5] = {global_dim3, global_dim2, global_dim1,
                                 global_dim0, 1};
  uint64_t gmem_prob_stride[5] = {
      static_cast<uint64_t>(BPE), static_cast<uint64_t>(BPE * global_dim3),
      static_cast<uint64_t>(BPE * global_dim3 * global_dim2),
      static_cast<uint64_t>(BPE * global_dim3 * global_dim2 * global_dim1), 0};
  uint32_t smem_box_shape[5] = {
      static_cast<uint32_t>(Dim3), static_cast<uint32_t>(Dim2),
      static_cast<uint32_t>(Dim1), static_cast<uint32_t>(Dim0), 1};
  uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

  CHECK_CUDA_DRIVER_ERROR(cuTensorMapEncodeTiled(
      &template_tensor_map, DataType, 4, gmem_address, gmem_prob_shape,
      gmem_prob_stride + 1, smem_box_shape, smem_box_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, Swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  return template_tensor_map;
}

//===----------------------------------------------------------------------===//
// Swizzle
//===----------------------------------------------------------------------===//

template <CUtensorMapSwizzle Swizzle = CU_TENSOR_MAP_SWIZZLE_NONE>
struct SwizzleIndexMap {
  __forceinline__ static __device__ int get_row(int row, int col, int bpe) {
    return row;
  }

  __forceinline__ static __device__ int get_col(int row, int col, int bpe) {
    return col;
  }
};

template <> struct SwizzleIndexMap<CU_TENSOR_MAP_SWIZZLE_128B> {
  __forceinline__ static __device__ int get_row(int row, int col, int bpe) {
    return row;
  }

  __forceinline__ static __device__ int get_col(int row, int col, int bpe) {
    int int4_row = row;
    int int4_col = col * bpe / sizeof(int4);
    return ((int4_row % 8) ^ int4_col) * sizeof(int4) / bpe;
  }
};

template <> struct SwizzleIndexMap<CU_TENSOR_MAP_SWIZZLE_64B> {
  __forceinline__ static __device__ int get_row(int row, int col, int bpe) {
    return row;
  }

  __forceinline__ static __device__ int get_col(int row, int col, int bpe) {
    int new_row = row * 64 / 128;
    int new_col = col * bpe / sizeof(int4);
    int row_swiz = new_row;
    int col_swiz = (row_swiz % 4) ^ new_col;
    return col_swiz * sizeof(int4) / bpe;
  }
};

//===----------------------------------------------------------------------===//
// Mbarrier
//===----------------------------------------------------------------------===//

__device__ static __forceinline__ void init_barrier(uint64_t *bar, int count) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
               "r"(count));
}

__device__ static __forceinline__ void expect_bytes(uint64_t *bar,
                                                    uint32_t bytes) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n" ::
          "r"(bar_ptr),
      "r"(bytes));
}

__device__ static __forceinline__ void wait(uint64_t *bar, int phase) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n"
      "@P1                       bra.uni DONE;\n"
      "bra.uni                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(mbar_ptr),
      "r"(phase));
}

__device__ static __forceinline__ void arrive(uint64_t *bar,
                                              uint32_t count = 1) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
               :
               : "r"(mbar_ptr), "r"(count)
               : "memory");
}

//===----------------------------------------------------------------------===//
// TMA with mbarrier
//===----------------------------------------------------------------------===//

__device__ static __forceinline__ void
load_4d_async(void *dst, void const *const tma_map, uint64_t *bar, int off0,
              int off1, int off2, int off3) {
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(tma_map);
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

  asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::"
               "complete_tx::bytes"
               " [%0], [%1, {%3, %4, %5, %6}], [%2];"
               :
               : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(off3),
                 "r"(off2), "r"(off1), "r"(off0)
               : "memory");
}

//===----------------------------------------------------------------------===//
// Convert to FP8E4M3
//===----------------------------------------------------------------------===//

template <int N0, int N1>
static __device__ __forceinline__ void
convert_to_fp8(const DFrag_F32_16x8 (&p_f32_in)[N0][N1],
               AFrag_FP8_16x32 (&p_fp8_out)[N0][8 * N1 / 32]) {
#pragma unroll
  for (int n0 = 0; n0 < N0; ++n0) {
#pragma unroll
    for (int i = 0; i < N1; i += 2) {
      uint32_t sfu32[2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        F8E4M3x4 out_f8x4 = fp32x4_to_e4m3x4(
            p_f32_in[n0][i].data[j * 2], p_f32_in[n0][i].data[j * 2 + 1],
            p_f32_in[n0][i + 1].data[j * 2],
            p_f32_in[n0][i + 1].data[j * 2 + 1]);
        p_fp8_out[n0][i * 8 / 32].data[j + 2 * ((i / 2) % 2)] = out_f8x4;
      } // end loop j
    } // end loop i
  } // end loop n0

  return;
}

template <typename Policy>
__launch_bounds__(Policy::THREAD_NUM, 1) __global__
    void qk_int8_pv_fp8_mha_fwd_sm120_kernel(
        BF16 *__restrict__ O, int64_t o_b_stride, int64_t o_h_stride,
        int64_t o_n_stride, int64_t o_d_stride, float *__restrict__ Q_SF,
        int64_t q_sf_b_stride, int64_t q_sf_h_stride, int64_t q_sf_n_stride,
        float *__restrict__ K_SF, int64_t k_sf_b_stride, int64_t k_sf_h_stride,
        int64_t k_sf_n_stride, int64_t B, int64_t H, int64_t Nq, int64_t Nkv,
        int64_t Dqk, int64_t Dv, float sm_scale,
        const __grid_constant__ CUtensorMap q_tensor_map,
        const __grid_constant__ CUtensorMap k_tensor_map,
        const __grid_constant__ CUtensorMap v_tensor_map,
        const __grid_constant__ CUtensorMap v_sf_tensor_map) {
  int block_b = blockIdx.z;                            // batch
  int block_h = blockIdx.y;                            // head
  int block_n = blockIdx.x * Policy::TILE_DOT_BLOCK_M; // seq

  int tid = threadIdx.x;
  int lane_id = tid % 32;
  int warp_id = tid / 32;
  bool is_consumer = tid < Policy::CONSUMER_THREAD_NUM;
  bool is_producer = !is_consumer;

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
  int warp_dot0_m = warp_id * Policy::TILE_DOT0_WARP_M;
  int warp_dot0_n = 0;
  int warp_dot1_m = warp_id * Policy::TILE_DOT1_WARP_M;
  int warp_dot1_n = 0;

  //===--------------------------------------------------------------------===//
  // Define shared memory for q, k and v
  //===--------------------------------------------------------------------===//

  extern __shared__ __align__(1024) char smem[];

  //===--------------------------------------------------------------------===//
  // Define brrier for a and b tma
  //===--------------------------------------------------------------------===//

  __shared__ __align__(8) uint64_t empty_bar[Policy::STAGE];
  __shared__ __align__(8) uint64_t full_bar[Policy::STAGE];
  if (is_producer && lane_id == 0) {
#pragma unroll
    for (int i = 0; i < Policy::STAGE; ++i) {
      init_barrier(&empty_bar[i], Policy::CONSUMER_WARP_NUM);
      init_barrier(&full_bar[i], 1);
    }
  }
  asm volatile("barrier.cluster.arrive;\n" : :);
  asm volatile("barrier.cluster.wait;\n" : :);

  //===--------------------------------------------------------------------===//
  // Producer
  //===--------------------------------------------------------------------===//

  if (is_producer) {
    reg_dealloc<24>();
    if (tid == Policy::CONSUMER_THREAD_NUM) {
      int stage = 0;
      int phase = 0;

      /// Leading TMA Q/K/V and Q_SF/K_SF/V_SF

      /// Acquire
      wait(&empty_bar[stage], phase);
      load_4d_async(Policy::get_q_stage_mem(smem, stage), &q_tensor_map,
                    &full_bar[stage], block_b, block_h, block_dot_m, 0);
      load_4d_async(Policy::get_k_stage_mem(smem, stage), &k_tensor_map,
                    &full_bar[stage], block_b, block_h, 0, 0);
      load_4d_async(Policy::get_v_stage_mem(smem, stage), &v_tensor_map,
                    &full_bar[stage], block_b, block_h, 0, 0);
      load_4d_async(Policy::get_v_sf_stage_mem(smem, stage), &v_sf_tensor_map,
                    &full_bar[stage], 0, block_b, block_h, 0);
      /// Commit
      expect_bytes(&full_bar[stage], Policy::TILE_Q_BLOCK_SIZE +
                                         Policy::TILE_K_BLOCK_SIZE +
                                         Policy::TILE_V_BLOCK_SIZE +
                                         Policy::TILE_V_SF_BLOCK_SIZE);

      ++stage;

      /// Next
      for (int dot0_n = Policy::TILE_DOT0_BLOCK_N; dot0_n < DOT0_N;
           dot0_n += Policy::TILE_DOT0_BLOCK_N, ++stage) {
        if (stage == Policy::STAGE) {
          stage = 0;
          phase ^= 1;
        }

        /// Acquire
        wait(&empty_bar[stage], phase);

        /// TMA K/V and K_SF/V_SF
        load_4d_async(Policy::get_k_stage_mem(smem, stage), &k_tensor_map,
                      &full_bar[stage], block_b, block_h, dot0_n, 0);
        load_4d_async(Policy::get_v_stage_mem(smem, stage), &v_tensor_map,
                      &full_bar[stage], block_b, block_h, 0, dot0_n);

        /// Commit
        expect_bytes(&full_bar[stage],
                     Policy::TILE_K_BLOCK_SIZE + Policy::TILE_V_BLOCK_SIZE);
      }
    }
  } // end if (is_producer)

  //===--------------------------------------------------------------------===//
  // Consumer
  //===--------------------------------------------------------------------===//

  else {
    reg_alloc<240>();
    if (lane_id == 0) {
#pragma unroll
      for (int i = 0; i < Policy::STAGE; ++i) {
        arrive(&empty_bar[i], 1);
      }
    }

    /// Scale for softmax
    sm_scale = sm_scale * LOG2E_F;

    //===------------------------------------------------------------------===//
    // Define fragment
    //===------------------------------------------------------------------===//

    /// Q
    AFrag_I8_16x32
        q_frag[Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M]
              [Policy::TILE_DOT0_WARP_K / Policy::TILE_DOT0_ATOMIC_K];
    float q_sf_frag = Q_SF[block_b * q_sf_b_stride + block_h * q_sf_h_stride +
                           (block_n + warp_dot0_m) /
                               Policy::Q_QUANTIZE_BLOCK_SIZE * q_sf_n_stride];
    /// max(row max of S=Q @ K)
    float max0[Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M];
    float max1[Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M];
    /// l（sum of exp)
    float l0[Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M];
    float l1[Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M];
#pragma unroll
    for (int i = 0; i < Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M;
         ++i) {
      max0[i] = -INFINITY;
      max1[i] = -INFINITY;
      l0[i] = 0.0f;
      l1[i] = 0.0f;
    }
    /// o: P @ V
    DFrag_F32_16x8
        o_frag[Policy::TILE_DOT1_WARP_M / Policy::TILE_DOT1_ATOMIC_M]
              [Policy::TILE_DOT1_WARP_N / Policy::TILE_DOT1_ATOMIC_N];
#pragma unroll
    for (int i = 0; i < Policy::TILE_DOT1_WARP_M / Policy::TILE_DOT1_ATOMIC_M;
         ++i) {
#pragma unroll
      for (int j = 0; j < Policy::TILE_DOT1_WARP_N / Policy::TILE_DOT1_ATOMIC_N;
           ++j) {
#pragma unroll
        for (int k = 0; k < o_frag[i][j].REGISTERS_PER_THREAD; ++k) {
          /// Clear o fragment
          o_frag[i][j].data[k] = 0.0f;
        }
      }
    }

    for (int dot0_n = 0, stage = 0, phase = 0; dot0_n < DOT0_N;
         dot0_n += Policy::TILE_DOT0_BLOCK_N, ++stage) {
      if (stage == Policy::STAGE) {
        stage = 0;
        phase ^= 1;
      }

      block_dot0_n = dot0_n;

      /// Wait
      wait(&full_bar[stage], phase);

      if (dot0_n == 0) {
        /// Load shared to fragment for Q/Q_SF operand

#pragma unroll
        for (int i = 0;
             i < Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M; ++i) {
#pragma unroll
          for (int j = 0;
               j < Policy::TILE_DOT0_WARP_K / Policy::TILE_DOT0_ATOMIC_K; ++j) {
            /// Q
            int *q_regs = (int *)q_frag[i][j].data;
            int q_shared_row =
                warp_dot0_m + i * Policy::TILE_DOT0_ATOMIC_M + (lane_id % 16);
            int q_shared_col =
                (j * Policy::TILE_DOT0_ATOMIC_K + (lane_id / 16) * 16);
            int q_shared_row_swiz = SwizzleIndexMap<Policy::SWIZZLE_Q>::get_row(
                q_shared_row, q_shared_col, sizeof(typename Policy::Q_TYPE));
            int q_shared_col_swiz = SwizzleIndexMap<Policy::SWIZZLE_Q>::get_col(
                q_shared_row, q_shared_col, sizeof(typename Policy::Q_TYPE));
            uint32_t q_addr = __cvta_generic_to_shared(
                Policy::get_q_stage_mem(smem, stage) +
                q_shared_row_swiz * Policy::TILE_DOT0_BLOCK_K +
                q_shared_col_swiz);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                         "{%0, %1, %2, %3}, [%4];"
                         : "=r"(q_regs[0]), "=r"(q_regs[1]), "=r"(q_regs[2]),
                           "=r"(q_regs[3])
                         : "r"(q_addr));
          } // end loop i
        } // end loop j
      } // end if (dot0_n == 0) leading

      /// Load K_SF
      float k_sf_frag =
          K_SF[block_b * k_sf_b_stride + block_h * k_sf_h_stride +
               dot0_n / Policy::K_QUANTIZE_BLOCK_SIZE * k_sf_n_stride];

      /// S: I32
      DFrag_I32_16x8
          s_frag_i32[Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M]
                    [Policy::TILE_DOT0_WARP_N / Policy::TILE_DOT0_ATOMIC_N];

      /// Step 0: S = Q @ K

#pragma unroll
      for (int dot0_k = 0; dot0_k < Policy::TILE_DOT0_BLOCK_K;
           dot0_k += Policy::TILE_DOT0_ATOMIC_K) {
        int dot0_atomic_k_cnt = dot0_k / Policy::TILE_DOT0_ATOMIC_K;
#pragma unroll
        for (int dot0_atomic_m = 0; dot0_atomic_m < Policy::TILE_DOT0_WARP_M;
             dot0_atomic_m += Policy::TILE_DOT0_ATOMIC_M) {
          int dot0_atomic_m_cnt = dot0_atomic_m / Policy::TILE_DOT0_ATOMIC_M;
#pragma unroll
          for (int dot0_atomic_n = 0; dot0_atomic_n < Policy::TILE_DOT0_WARP_N;
               dot0_atomic_n += Policy::TILE_DOT0_ATOMIC_N) {
            int dot0_atomic_n_cnt = dot0_atomic_n / Policy::TILE_DOT0_ATOMIC_N;
            /// Load K
            BFrag_I8_32x8 k_frag;
            {
              int *k_regs = (int *)k_frag.data;
              int k_shared_row = warp_dot0_n + dot0_atomic_n + (lane_id % 8);
              int k_shared_col = dot0_k + ((lane_id % 16) / 8) * 16;
              int k_shared_row_swiz =
                  SwizzleIndexMap<Policy::SWIZZLE_K>::get_row(
                      k_shared_row, k_shared_col,
                      sizeof(typename Policy::K_TYPE));
              int k_shared_col_swiz =
                  SwizzleIndexMap<Policy::SWIZZLE_K>::get_col(
                      k_shared_row, k_shared_col,
                      sizeof(typename Policy::K_TYPE));
              uint32_t k_addr = __cvta_generic_to_shared(
                  Policy::get_k_stage_mem(smem, stage) +
                  k_shared_row_swiz * Policy::TILE_DOT0_BLOCK_K +
                  k_shared_col_swiz);
              asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                           "{%0, %1}, [%2];"
                           : "=r"(k_regs[0]), "=r"(k_regs[1])
                           : "r"(k_addr));
            }

            /// Dot0: Q @ V
            if (dot0_k == 0) {
              /// Init with zeor
              mma_sync_m16n8k32_row_col_i8i8i32<MMAAccumulateMode::kInit>(
                  reinterpret_cast<int32_t *>(
                      s_frag_i32[dot0_atomic_m_cnt][dot0_atomic_n_cnt].data),
                  q_frag[dot0_atomic_m_cnt][dot0_atomic_k_cnt].data,
                  k_frag.data);
            } else {
              /// Inplace s_frag
              mma_sync_m16n8k32_row_col_i8i8i32(
                  reinterpret_cast<int32_t *>(
                      s_frag_i32[dot0_atomic_m_cnt][dot0_atomic_n_cnt].data),
                  q_frag[dot0_atomic_m_cnt][dot0_atomic_k_cnt].data,
                  k_frag.data);
            }
          } // end loop dot0_atomic_n
        } // end loop dot0_atomic_m
      } // end loop dot0_k

      /// Convert s_frag_i32 to float
      DFrag_F32_16x8
          s_frag[Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M]
                [Policy::TILE_DOT0_WARP_N / Policy::TILE_DOT0_ATOMIC_N];

      float qk_scale = sm_scale * q_sf_frag * k_sf_frag;

#pragma unroll
      for (int i = 0; i < Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M;
           ++i) {
        int atomic_m = i * Policy::TILE_DOT0_ATOMIC_M;
#pragma unroll
        for (int j = 0;
             j < Policy::TILE_DOT0_WARP_N / Policy::TILE_DOT0_ATOMIC_N; ++j) {
          int atomic_n = j * Policy::TILE_DOT0_ATOMIC_N;

          auto *in_i32 = reinterpret_cast<int32_t *>(s_frag_i32[i][j].data);
          auto *out_f32 = reinterpret_cast<float *>(s_frag[i][j].data);

          out_f32[0] = __int2float_rz(in_i32[0]) * qk_scale;
          out_f32[1] = __int2float_rz(in_i32[1]) * qk_scale;
          out_f32[2] = __int2float_rz(in_i32[2]) * qk_scale;
          out_f32[3] = __int2float_rz(in_i32[3]) * qk_scale;
        }
      }

      /// mask to inf
      if (dot0_n + Policy::TILE_DOT0_BLOCK_N > DOT0_N) {
#pragma unroll
        for (int dot0_atomic_n = 0; dot0_atomic_n < Policy::TILE_DOT0_WARP_N;
             dot0_atomic_n += Policy::TILE_DOT0_ATOMIC_N) {
          int j = dot0_atomic_n / Policy::TILE_DOT0_ATOMIC_N;
#pragma unroll
          for (int k = 0; k < s_frag[0][j].REGISTERS_PER_THREAD; ++k) {
            int thread_n = s_frag[0][j].get_col_with_reg(lane_id, k);
            if (dot0_atomic_n + thread_n + dot0_n >= DOT0_N) {
#pragma unroll
              for (int dot0_atomic_m = 0;
                   dot0_atomic_m < Policy::TILE_DOT0_WARP_M;
                   dot0_atomic_m += Policy::TILE_DOT0_ATOMIC_M) {
                int i = dot0_atomic_m / Policy::TILE_DOT0_ATOMIC_M;
                s_frag[i][j].data[k] = -INFINITY;
              }
            }
          }
        }
      }

      /// Step 2: max = rowmax(S)
      float max_new0[Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M];
      float max_new1[Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M];
#pragma unroll
      for (int i = 0; i < Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M;
           ++i) {
        auto *data = reinterpret_cast<float *>(s_frag[i][0].data);
        max_new0[i] = fmaxf(data[0], data[1]);
        max_new1[i] = fmaxf(data[2], data[3]);
#pragma unroll
        for (int j = 1;
             j < Policy::TILE_DOT0_WARP_N / Policy::TILE_DOT0_ATOMIC_N; ++j) {
          auto *data = reinterpret_cast<float *>(s_frag[i][j].data);

          max_new0[i] = fmaxf(max_new0[i], data[0]);
          max_new0[i] = fmaxf(max_new0[i], data[1]);
          max_new1[i] = fmaxf(max_new1[i], data[2]);
          max_new1[i] = fmaxf(max_new1[i], data[3]);
        } // end loop j
        max_new0[i] =
            fmaxf(__shfl_xor_sync(0xffffffff, max_new0[i], 1), max_new0[i]);
        max_new0[i] =
            fmaxf(__shfl_xor_sync(0xffffffff, max_new0[i], 2), max_new0[i]);
        max_new1[i] =
            fmaxf(__shfl_xor_sync(0xffffffff, max_new1[i], 1), max_new1[i]);
        max_new1[i] =
            fmaxf(__shfl_xor_sync(0xffffffff, max_new1[i], 2), max_new1[i]);

        max_new0[i] -= Policy::EXP_OFFSET;
        max_new1[i] -= Policy::EXP_OFFSET;

        max_new0[i] = fmaxf(max_new0[i], max0[i]);
        max_new1[i] = fmaxf(max_new1[i], max1[i]);
      } // end loop i

      /// Step 3: p = exp(S - max)
      DFrag_F32_16x8
          p_frag[Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M]
                [Policy::TILE_DOT0_WARP_N / Policy::TILE_DOT0_ATOMIC_N];
#pragma unroll
      for (int i = 0; i < Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M;
           ++i) {
#pragma unroll
        for (int j = 0;
             j < Policy::TILE_DOT0_WARP_N / Policy::TILE_DOT0_ATOMIC_N; ++j) {
          auto *data = reinterpret_cast<float *>(s_frag[i][j].data);

          p_frag[i][j].data[0] = exp2f(data[0] - max_new0[i]);
          p_frag[i][j].data[1] = exp2f(data[1] - max_new0[i]);
          p_frag[i][j].data[2] = exp2f(data[2] - max_new1[i]);
          p_frag[i][j].data[3] = exp2f(data[3] - max_new1[i]);
        } // end loop j
      } // end loop i

      /// Step 4: l = sum(p)
      float l_new0[Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M];
      float l_new1[Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M];
#pragma unroll
      for (int i = 0; i < Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M;
           ++i) {
        l_new0[i] = p_frag[i][0].data[0] + p_frag[i][0].data[1];
        l_new1[i] = p_frag[i][0].data[2] + p_frag[i][0].data[3];
#pragma unroll
        for (int j = 1;
             j < Policy::TILE_DOT0_WARP_N / Policy::TILE_DOT0_ATOMIC_N; ++j) {
          l_new0[i] += p_frag[i][j].data[0];
          l_new0[i] += p_frag[i][j].data[1];
          l_new1[i] += p_frag[i][j].data[2];
          l_new1[i] += p_frag[i][j].data[3];
        } // end loop j
        l_new0[i] += __shfl_xor_sync(0xffffffff, l_new0[i], 1);
        l_new0[i] += __shfl_xor_sync(0xffffffff, l_new0[i], 2);
        l_new1[i] += __shfl_xor_sync(0xffffffff, l_new1[i], 1);
        l_new1[i] += __shfl_xor_sync(0xffffffff, l_new1[i], 2);
      } // end loop i

      /// max scale
      float max_scale0[Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M];
      float max_scale1[Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M];
#pragma unroll
      for (int i = 0; i < Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M;
           ++i) {
        max_scale0[i] = exp2f(max0[i] - max_new0[i]);
        max_scale1[i] = exp2f(max1[i] - max_new1[i]);
      }

      /// update max and sumexp
#pragma unroll
      for (int i = 0; i < Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M;
           ++i) {
        max0[i] = max_new0[i];
        max1[i] = max_new1[i];
        l0[i] = fmaf(l0[i], max_scale0[i], l_new0[i]);
        l1[i] = fmaf(l1[i], max_scale1[i], l_new1[i]);
      }

      /// Step 5: o = P @ V
      AFrag_FP8_16x32
          p_fp8_frag[Policy::TILE_DOT1_WARP_M / Policy::TILE_DOT1_ATOMIC_M]
                    [Policy::TILE_DOT1_WARP_K / Policy::TILE_DOT1_ATOMIC_K];
      BFrag_FP8_32x8
          v_frag[Policy::TILE_DOT1_WARP_N / Policy::TILE_DOT1_ATOMIC_N]
                [Policy::TILE_DOT1_WARP_K / Policy::TILE_DOT1_ATOMIC_K];

#pragma unroll
      for (int dot1_atomic_m = 0; dot1_atomic_m < Policy::TILE_DOT1_WARP_M;
           dot1_atomic_m += Policy::TILE_DOT1_ATOMIC_M) {
        int dot1_atomic_m_cnt = dot1_atomic_m / Policy::TILE_DOT1_ATOMIC_M;

#pragma unroll
        for (int dot1_atomic_n = 0; dot1_atomic_n < Policy::TILE_DOT1_WARP_N;
             dot1_atomic_n += Policy::TILE_DOT1_ATOMIC_N) {
          int dot1_atomic_n_cnt = dot1_atomic_n / Policy::TILE_DOT1_ATOMIC_N;

          DFrag_F16_16x8 o_block_f16;

#pragma unroll
          for (int dot1_atomic_k = 0; dot1_atomic_k < Policy::TILE_DOT1_BLOCK_K;
               dot1_atomic_k += Policy::TILE_DOT1_ATOMIC_K) {
            int dot1_atomic_k_cnt = dot1_atomic_k / Policy::TILE_DOT1_ATOMIC_K;

            /// quantize P from f32 to fp8e4m3
            if (dot1_atomic_n == 0) {
              AFrag_FP8_16x32(&p_fp8_frag_v1)[1][1] =
                  reinterpret_cast<AFrag_FP8_16x32(&)[1][1]>(
                      p_fp8_frag[dot1_atomic_m_cnt][dot1_atomic_k_cnt]);
              constexpr int SIZE0 =
                  Policy::TILE_DOT0_WARP_M / Policy::TILE_DOT0_ATOMIC_M *
                  Policy::TILE_DOT0_WARP_N / Policy::TILE_DOT0_ATOMIC_N / 4;
              constexpr int SIZE1 = 4;
              DFrag_F32_16x8(&p_frag_v4)[SIZE0][SIZE1] =
                  reinterpret_cast<DFrag_F32_16x8(&)[SIZE0][SIZE1]>(p_frag);
              DFrag_F32_16x8(&p_frag_v4_this)[1][4] =
                  reinterpret_cast<DFrag_F32_16x8(&)[1][4]>(
                      p_frag_v4[(dot1_atomic_m / Policy::TILE_DOT0_ATOMIC_M *
                                     Policy::TILE_DOT0_WARP_N /
                                     Policy::TILE_DOT0_ATOMIC_N +
                                 dot1_atomic_k / Policy::TILE_DOT0_ATOMIC_N) /
                                4]);
              convert_to_fp8(p_frag_v4_this, p_fp8_frag_v1);
            }

            /// load v
            if (dot1_atomic_m == 0) {
              int *v_regs =
                  (int *)v_frag[dot1_atomic_n_cnt][dot1_atomic_k_cnt].data;
              int v_shared_row = warp_dot1_n + dot1_atomic_n + (lane_id % 8);
              int v_shared_col = dot1_atomic_k + ((lane_id % 16) / 8) * 16;
              int v_shared_row_swiz =
                  SwizzleIndexMap<Policy::SWIZZLE_V>::get_row(
                      v_shared_row, v_shared_col,
                      sizeof(typename Policy::V_TYPE));
              int v_shared_col_swiz =
                  SwizzleIndexMap<Policy::SWIZZLE_V>::get_col(
                      v_shared_row, v_shared_col,
                      sizeof(typename Policy::V_TYPE));
              uint32_t v_addr = __cvta_generic_to_shared(
                  Policy::get_v_stage_mem(smem, stage) +
                  v_shared_row_swiz * Policy::TILE_DOT1_BLOCK_K +
                  v_shared_col_swiz);
              asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                           "{%0, %1}, [%2];"
                           : "=r"(v_regs[0]), "=r"(v_regs[1])
                           : "r"(v_addr));
            }

            /// P @ V
            if (dot1_atomic_k == 0) {
              mma_sync_m16n8k32_row_col_fp8fp8f16<MMAAccumulateMode::kInit>(
                  o_block_f16.data,
                  p_fp8_frag[dot1_atomic_m_cnt][dot1_atomic_k_cnt].data,
                  v_frag[dot1_atomic_n_cnt][dot1_atomic_k_cnt].data);
            } else {
              mma_sync_m16n8k32_row_col_fp8fp8f16(
                  o_block_f16.data,
                  p_fp8_frag[dot1_atomic_m_cnt][dot1_atomic_k_cnt].data,
                  v_frag[dot1_atomic_n_cnt][dot1_atomic_k_cnt].data);
            }
          } // end loop dot1_atomic_k

          /// update o = o + f32(o_block_f16)

          __half *in_u16 = reinterpret_cast<__half *>(o_block_f16.data);
          float *o_f32 = o_frag[dot1_atomic_m_cnt][dot1_atomic_n_cnt].data;
          o_f32[0] = fmaf(o_f32[0], max_scale0[dot1_atomic_m_cnt],
                          __half2float(in_u16[0]));
          o_f32[1] = fmaf(o_f32[1], max_scale0[dot1_atomic_m_cnt],
                          __half2float(in_u16[1]));
          o_f32[2] = fmaf(o_f32[2], max_scale1[dot1_atomic_m_cnt],
                          __half2float(in_u16[2]));
          o_f32[3] = fmaf(o_f32[3], max_scale1[dot1_atomic_m_cnt],
                          __half2float(in_u16[3]));
        } // end loop dot1_atomic_n
      } // end loop dot1_atomic_m

      /// Release
      if (lane_id == 0) {
        arrive(&empty_bar[stage], 1);
      }
    } // end loop dot0_n/dot1_k

    //===------------------------------------------------------------------===//
    // Epilogue
    //===------------------------------------------------------------------===//

    /// step 8: o = o / l * v_scale

#pragma unroll
    for (int i = 0; i < Policy::TILE_DOT1_WARP_M / Policy::TILE_DOT1_ATOMIC_M;
         ++i) {
      float scale0 = 1.0f / l0[i];
      float scale1 = 1.0f / l1[i];

#pragma unroll
      for (int j = 0; j < Policy::TILE_DOT1_WARP_N / Policy::TILE_DOT1_ATOMIC_N;
           ++j) {
        int thread_n = (lane_id % 4) * 2;
        int n = warp_dot1_n + j * Policy::TILE_DOT1_ATOMIC_N + thread_n;
        float v_scale0 = Policy::get_v_sf_stage_mem(smem, 0)[n];
        float v_scale1 = Policy::get_v_sf_stage_mem(smem, 0)[n + 1];
        o_frag[i][j].data[0] *= scale0 * v_scale0;
        o_frag[i][j].data[1] *= scale0 * v_scale1;
        o_frag[i][j].data[2] *= scale1 * v_scale0;
        o_frag[i][j].data[3] *= scale1 * v_scale1;
      }
    }

    /// Step 9: Write back to O

    if (block_dot_m + warp_dot0_m + Policy::TILE_DOT0_WARP_M <= Nq &&
        block_dot_n + warp_dot0_n + Policy::TILE_DOT1_WARP_N <= Dv) {
#pragma unroll
      for (int i = 0; i < Policy::TILE_DOT1_WARP_M / Policy::TILE_DOT1_ATOMIC_M;
           ++i) {
#pragma unroll
        for (int j = 0;
             j < Policy::TILE_DOT1_WARP_N / Policy::TILE_DOT1_ATOMIC_N; ++j) {
#pragma unroll
          for (int idx = 0; idx < o_frag[i][j].REGISTERS_PER_THREAD; ++idx) {
            int thread_m = i * Policy::TILE_DOT1_ATOMIC_M +
                           o_frag[i][j].get_row_with_reg(lane_id, idx);
            // map m to O's N(seq)
            int m = block_dot_m + warp_dot1_m + thread_m;
            int thread_n = j * Policy::TILE_DOT1_ATOMIC_N +
                           o_frag[i][j].get_col_with_reg(lane_id, idx);
            // map n to O's D(emb)
            int n = block_dot_n + warp_dot1_n + thread_n;
            auto *o_tile = O + block_b * o_b_stride + m * o_n_stride +
                           block_h * o_h_stride + n * o_d_stride;
            *o_tile = __float2bfloat16_rn(o_frag[i][j].data[idx]);
          }
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < Policy::TILE_DOT1_WARP_M / Policy::TILE_DOT1_ATOMIC_M;
           ++i) {
#pragma unroll
        for (int j = 0;
             j < Policy::TILE_DOT1_WARP_N / Policy::TILE_DOT1_ATOMIC_N; ++j) {
#pragma unroll
          for (int idx = 0; idx < o_frag[i][j].REGISTERS_PER_THREAD; ++idx) {
            int thread_m = i * Policy::TILE_DOT1_ATOMIC_M +
                           o_frag[i][j].get_row_with_reg(lane_id, idx);
            // map m to O's N(seq)
            int m = block_dot_m + warp_dot1_m + thread_m;
            int thread_n = j * Policy::TILE_DOT1_ATOMIC_N +
                           o_frag[i][j].get_col_with_reg(lane_id, idx);
            // map n to O's D(emb)
            int n = block_dot_n + warp_dot1_n + thread_n;
            if (m < Nq && n < Dv) {
              auto *o_tile = O + block_b * o_b_stride + m * o_n_stride +
                             block_h * o_h_stride + n * o_d_stride;
              *o_tile = __float2bfloat16_rn(o_frag[i][j].data[idx]);
            }
          }
        }
      }
    }
  } // end if(is_consumer)
}

//===--------------------------------------------------------------------===//
// ac4k mha attention with int8 & fp8 acceleration
// Q               :[B,   H,    Nq,     Dqk]xINT8
// K               :[B,   H,    Nkv,    Dqk]xINT8
// V               :[B,   H,    Dv,     Nkv]xFP8E4M3
// O               :[B,   H,    Nq,     Dv]xBF16
// Q_SF            :[B,   H,    Nq]xF32
// K_SF            :[B,   H,    Nkv]xF32
// V_SF            :[B,   H,    Dv]xF32
// Dqk             : Head dim size for Q and K
//===--------------------------------------------------------------------===//

void qk_int8_pv_fp8_mha_fwd_sm120(torch::Tensor &o, torch::Tensor &q,
                                  torch::Tensor &q_sf,
                                  int q_quantize_block_size, torch::Tensor &k,
                                  torch::Tensor &k_sf,
                                  int k_quantize_block_size, torch::Tensor &v,
                                  torch::Tensor &v_sf, float sm_scale) {
  /// CHECK Q & Q_SF
  CHECK_INPUT(q, at::ScalarType::Char, "Q must be pack to int8 tensor");
  TORCH_CHECK(q.dim() == 4, "Q must be a 4D tensor");
  int64_t B = q.size(0);
  int64_t H = q.size(1);
  int64_t Nq = q.size(2);
  int64_t Dqk = q.size(3);
  TORCH_CHECK(Dqk % 16 == 0, "Dqk must be multiple of 16");
  /// TODO(need remove limit)
  TORCH_CHECK(Dqk <= MAX_HEAD_DIM_SIZE, "Dqk must be less than ",
              MAX_HEAD_DIM_SIZE);
  CHECK_INPUT(q_sf, at::ScalarType::Float, "Q_SF must be a float tensor");
  TORCH_CHECK(q_sf.dim() == 3, "Q_SF must be a 3D tensor");
  TORCH_CHECK(q_sf.size(0) == B, "Meet invalid Q_SF size(0) with ",
              q_sf.size(0), "==", B);
  TORCH_CHECK(q_sf.size(1) == H, "Meet invalid Q_SF size(1) with ",
              q_sf.size(1), "==", H);
  TORCH_CHECK(q_sf.size(2) == ceil_div(Nq, q_quantize_block_size),
              "Meet invalid Q_SF size(2)");

  /// CHECK K & K_SF
  CHECK_INPUT(k, at::ScalarType::Char, "K must be pack to int8 tensor");
  TORCH_CHECK(k.dim() == 4, "K must be a 4D tensor");
  TORCH_CHECK(k.size(0) == B, "K must have the same batch size as Q");
  TORCH_CHECK(k.size(1) == H, "K must have the same head number as Q");
  int64_t Nkv = k.size(2);
  TORCH_CHECK(k.size(3) == q.size(3), "K must have the same dim as Q");
  CHECK_INPUT(k_sf, at::ScalarType::Float, "K_SF must be a float tensor");
  TORCH_CHECK(k_sf.dim() == 3, "K_SF must be a 3D tensor");
  TORCH_CHECK(k_sf.size(0) == B, "Meet invalid K_SF size(0) with ",
              k_sf.size(0), "==", B);
  TORCH_CHECK(k_sf.size(1) == H, "Meet invalid K_SF size(1) with ",
              k_sf.size(1), "==", H);
  TORCH_CHECK(k_sf.size(2) == ceil_div(Nkv, k_quantize_block_size),
              "Meet invalid K_SF size(2)");

  /// CHECK V & V_SF
  CHECK_INPUT(v, at::ScalarType::Float8_e4m3fn,
              "V must be pack to f8e4m3 tensor");
  TORCH_CHECK(v.dim() == 4, "V must be a 4D tensor");
  TORCH_CHECK(v.size(0) == B, "V must have the same batch size as Q");
  TORCH_CHECK(v.size(1) == H, "V must have the same head number as Q");
  int64_t Dv = v.size(2);
  TORCH_CHECK(v.size(3) == align_up(Nkv, 16), "Meet invalid v size(3) with ",
              v.size(3), "==", align_up(Nkv, 16));
  /// TODO(need remove limit)
  TORCH_CHECK(Dv <= MAX_HEAD_DIM_SIZE, "Dqk must be less than ",
              MAX_HEAD_DIM_SIZE);
  CHECK_INPUT(v_sf, at::ScalarType::Float, "V_SF must be a float tensor");
  TORCH_CHECK(v_sf.dim() == 3, "V_SF must be a 3D tensor");
  TORCH_CHECK(v_sf.size(0) == B, "Meet invalid v_sf size(0) with ",
              v_sf.size(0), "==", B);
  TORCH_CHECK(v_sf.size(1) == H, "Meet invalid v_sf size(1) with ",
              v_sf.size(1), "==", H);
  TORCH_CHECK(v_sf.size(2) == align_up(Dv, 16),
              "Meet invalid v_sf size(2) with ", v_sf.size(2),
              "==", align_up(Dv, 16));

  /// CHECK O
  CHECK_OUTPUT(o, at::ScalarType::BFloat16, "O must be a bfloat16 tensor");
  TORCH_CHECK(o.dim() == 4, "O must be a 4D tensor");
  TORCH_CHECK(o.size(0) == B, "O must have the same batch size as Q");
  TORCH_CHECK(o.size(1) == H, "O must have the same head number as Q");
  TORCH_CHECK(o.size(2) == Nq, "O must have the same sequence length as Q");
  TORCH_CHECK(o.size(3) == Dv, "O must have the same dim as V");

  DISPATCH_VALUE<int, 32,
                 64>(q_quantize_block_size, [&]<int Q_QUANTIZE_BLOCK_SIZE>() {
    DISPATCH_VALUE<
        int, 128>(k_quantize_block_size, [&]<int K_QUANTIZE_BLOCK_SIZE>() {
      DISPATCH_HEAD_DIM_SIZES<
          64, 128>(static_cast<int>(Dqk), [&]<int HEAD_DIM_QK>() {
        DISPATCH_HEAD_DIM_SIZES<
            64, 128>(static_cast<int>(Dv), [&]<int HEAD_DIM_V>() {
          /// Policy
          using policy = Policy<HEAD_DIM_QK, HEAD_DIM_V, Q_QUANTIZE_BLOCK_SIZE,
                                K_QUANTIZE_BLOCK_SIZE>;

          static_assert(Q_QUANTIZE_BLOCK_SIZE % policy::TILE_DOT0_WARP_M == 0,
                        "invalid q_quantize_block_size");
          static_assert(K_QUANTIZE_BLOCK_SIZE % policy::TILE_DOT0_WARP_N == 0,
                        "invalid k_quantize_block_size");

          /// TMA descriptor
          CUtensorMap q_tensor_map = create_4d_tensor_map<
              1, 1, policy::TILE_DOT0_BLOCK_M, policy::TILE_DOT0_BLOCK_K,
              CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8, uint8_t,
              policy::SWIZZLE_Q>(
              reinterpret_cast<const uint8_t *>(q.data_ptr()),
              static_cast<uint64_t>(q.size(0)),
              static_cast<uint64_t>(q.size(1)),
              static_cast<uint64_t>(q.size(2)),
              static_cast<uint64_t>(q.size(3)));
          CUtensorMap k_tensor_map = create_4d_tensor_map<
              1, 1, policy::TILE_DOT0_BLOCK_N, policy::TILE_DOT0_BLOCK_K,
              CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8, uint8_t,
              policy::SWIZZLE_K>(
              reinterpret_cast<const uint8_t *>(k.data_ptr()),
              static_cast<uint64_t>(k.size(0)),
              static_cast<uint64_t>(k.size(1)),
              static_cast<uint64_t>(k.size(2)),
              static_cast<uint64_t>(k.size(3)));
          CUtensorMap v_tensor_map = create_4d_tensor_map<
              1, 1, policy::TILE_DOT1_BLOCK_N, policy::TILE_DOT1_BLOCK_K,
              CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8, F8E4M3,
              policy::SWIZZLE_V>(reinterpret_cast<const F8E4M3 *>(v.data_ptr()),
                                 static_cast<uint64_t>(v.size(0)),
                                 static_cast<uint64_t>(v.size(1)),
                                 static_cast<uint64_t>(v.size(2)),
                                 static_cast<uint64_t>(v.size(3)));
          CUtensorMap v_sf_tensor_map = create_4d_tensor_map<
              1, 1, 1, policy::TILE_DOT1_BLOCK_N,
              CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32, uint32_t>(
              reinterpret_cast<const uint32_t *>(v_sf.data_ptr()), 1UL,
              static_cast<uint64_t>(v_sf.size(0)),
              static_cast<uint64_t>(v_sf.size(1)),
              static_cast<uint64_t>(v_sf.size(2)));

          /// Set dynamic shared memory size
          auto *kernel = qk_int8_pv_fp8_mha_fwd_sm120_kernel<policy>;
          CHECK_CUDA_ERROR(cudaFuncSetAttribute(
              kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
              policy::SMEM_SIZE));

          /// Get CUDA stream
          auto stream = at::cuda::getCurrentCUDAStream().stream();

          /// Launch kernel
          dim3 grid(
              ceil_div(Nq, static_cast<int64_t>(policy::TILE_DOT_BLOCK_M)), H,
              B);
          dim3 block(policy::THREAD_NUM);
          kernel<<<grid, block, policy::SMEM_SIZE, stream>>>(
              reinterpret_cast<BF16 *>(o.data_ptr()),
              /* o b stride */ o.stride(0),
              /* o h stride */ o.stride(1),
              /* o n stride */ o.stride(2),
              /* o d stride */ o.stride(3),
              reinterpret_cast<float *>(q_sf.data_ptr()),
              /* q_sf b stride */ q_sf.stride(0),
              /* q_sf h stride */ q_sf.stride(1),
              /* q_sf n stride */ q_sf.stride(2),
              reinterpret_cast<float *>(k_sf.data_ptr()),
              /* k_sf b stride */ k_sf.stride(0),
              /* k_sf h stride */ k_sf.stride(1),
              /* k_sf n stride */ k_sf.stride(2), B, H, Nq, Nkv, Dqk, Dv,
              sm_scale, q_tensor_map, k_tensor_map, v_tensor_map,
              v_sf_tensor_map);
        }); // end of DISPATCH_HEAD_DIM_SIZES HEAD_DIM_V
      });   // end of DISPATCH_HEAD_DIM_SIZES HEAD_DIM_QK
    });     // end of DISPATCH_VALUE K_QUANTIZE_BLOCK_SIZE
  });       // end of DISPATCH_VALUE Q_QUANTIZE_BLOCK_SIZE
}

} // namespace ac4k
