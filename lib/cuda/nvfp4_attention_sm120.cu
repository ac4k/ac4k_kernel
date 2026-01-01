#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
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
#define CHECK_OUTPUT(x, st, m)                                                 \
  CHECK_TH_CUDA(x, m);                                                         \
  CHECK_TYPE(x, st, m)
#define CHECK_SCALAR(x, st, m)                                                 \
  CHECK_TH_CUDA(x, m);                                                         \
  CHECK_TYPE(x, st, m);                                                        \
  TORCH_CHECK(x.dim() == 0, m)

namespace ac4k {

using BF16 = __nv_bfloat16;
using NVFP4x2 = uint8_t;
using NVFP4x8 = uint32_t;
using F8E4M3 = uint8_t;
constexpr int ELES_PER_NVFP4x2 = 2;

constexpr int64_t HEAD_DIM_SIZE = 128;
constexpr int64_t BLOCK_SIZE = 16;
constexpr int64_t HEAD_DIM_ALIGN_SIZE = 128;
constexpr int CROSS_DIM_ALIGN_SIZE = 16;

//===----------------------------------------------------------------------===//
// Define tile size for block level, warp level and atomic level
//===----------------------------------------------------------------------===//

/// Block level tile size
constexpr int TILE_DOT_BLOCK_M = 128;
constexpr int TILE_DOT0_BLOCK_M = TILE_DOT_BLOCK_M;
constexpr int TILE_DOT0_BLOCK_N = 128;
constexpr int TILE_DOT0_BLOCK_K = 128;
constexpr int TILE_DOT1_BLOCK_M = TILE_DOT0_BLOCK_M;
constexpr int TILE_DOT1_BLOCK_N = 128;
constexpr int TILE_DOT1_BLOCK_K = TILE_DOT0_BLOCK_N;

/// Warp level tile size
constexpr int TILE_DOT0_WARP_M = 16;
constexpr int TILE_DOT0_WARP_N = TILE_DOT0_BLOCK_N;
constexpr int TILE_DOT0_WARP_K = TILE_DOT0_BLOCK_K;
constexpr int TILE_DOT1_WARP_M = TILE_DOT0_WARP_M;
constexpr int TILE_DOT1_WARP_N = 128;
constexpr int TILE_DOT1_WARP_K = TILE_DOT1_BLOCK_K;

constexpr int WARP_NUM =
    TILE_DOT0_BLOCK_M / TILE_DOT0_WARP_M * TILE_DOT0_BLOCK_N / TILE_DOT0_WARP_N;
constexpr int WARP_SIZE = 32;

/// Atomic level tile size
constexpr int TILE_ATOMIC_M = 16;
constexpr int TILE_ATOMIC_N = 8;
constexpr int TILE_ATOMIC_K = 64;

/// Tile size
constexpr int TILE_Q_BLOCK_ELES = TILE_DOT0_BLOCK_M * TILE_DOT0_BLOCK_K;
constexpr int TILE_K_BLOCK_ELES = TILE_DOT0_BLOCK_K * TILE_DOT0_BLOCK_N;
constexpr int TILE_V_BLOCK_ELES = TILE_DOT1_BLOCK_K * TILE_DOT1_BLOCK_N;
constexpr int TILE_Q_SF_BLOCK_ELES = TILE_Q_BLOCK_ELES / BLOCK_SIZE;
constexpr int TILE_K_SF_BLOCK_ELES = TILE_K_BLOCK_ELES / BLOCK_SIZE;
constexpr int TILE_V_SF_BLOCK_ELES = TILE_V_BLOCK_ELES / BLOCK_SIZE;
constexpr int TILE_Q_BLOCK_SIZE =
    TILE_Q_BLOCK_ELES / ELES_PER_NVFP4x2 * sizeof(NVFP4x2);
constexpr int TILE_K_BLOCK_SIZE =
    TILE_K_BLOCK_ELES / ELES_PER_NVFP4x2 * sizeof(NVFP4x2);
constexpr int TILE_V_BLOCK_SIZE =
    TILE_V_BLOCK_ELES / ELES_PER_NVFP4x2 * sizeof(NVFP4x2);
constexpr int TILE_Q_SF_BLOCK_SIZE = TILE_Q_SF_BLOCK_ELES * sizeof(F8E4M3);
constexpr int TILE_K_SF_BLOCK_SIZE = TILE_K_SF_BLOCK_ELES * sizeof(F8E4M3);
constexpr int TILE_V_SF_BLOCK_SIZE = TILE_V_SF_BLOCK_ELES * sizeof(F8E4M3);

//===----------------------------------------------------------------------===//
// Define mutiple buffer stage
//===----------------------------------------------------------------------===//

const int STAGE = 2;

//===----------------------------------------------------------------------===//
// Consumer & Producer
//===----------------------------------------------------------------------===//

const int CONSUMER_THREAD_NUM = WARP_NUM * 32;
const int PRODUCER_THREAD_NUM = 128;

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
// Fast reciprocal
//===----------------------------------------------------------------------===//

__forceinline__ __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

//===----------------------------------------------------------------------===//
// Heterogeneous allocation of registers for producer & consumer.
//===----------------------------------------------------------------------===//

template <uint32_t RegCount> __forceinline__ __device__ void reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount> __forceinline__ __device__ void reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

//===----------------------------------------------------------------------===//
// Convert 8xfloat32 to 8xNVFP4 (represented as one uint32_t)
//===----------------------------------------------------------------------===//

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
  for (int n0 = 0; n0 < N0; ++n0) {
#pragma unroll
    for (int i = 0; i < N1; i += 8) {
      uint32_t sfu32[2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int k = 0; k < 8; k += 4) {
          float max = fmax(p_f32_in[n0][i + k].data[j * 2],
                           p_f32_in[n0][i + k].data[j * 2 + 1]);
#pragma unroll
          for (int l = 1; l < 4; ++l) {
            max = fmax(max, p_f32_in[n0][i + k + l].data[j * 2]);
            max = fmax(max, p_f32_in[n0][i + k + l].data[j * 2 + 1]);
          }
          max = fmax(__shfl_xor_sync(0xffffffff, max, 1), max);

          /// sf
          float sf_value =
              rcp_global_scale_value * (max * 0.16666666666666666f);
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

          float out_scale = sf_value != 0
                                ? rcp_global_scale_value *
                                      reciprocal_approximate_ftz(sf_value)
                                : 0.0f;

          float in[8];
#pragma unroll
          for (int m = 0; m < 4; ++m) {
            in[2 * m] = p_f32_in[n0][i + k + m].data[j * 2] * out_scale;
            in[2 * m + 1] = p_f32_in[n0][i + k + m].data[j * 2 + 1] * out_scale;
          }

          /// Convert 8xfloat32 to 8xe2m1(uint32)
          uint32_t e2m1x8 = fp32_vec_to_e2m1(in);
          p_fp4_out[n0][8 * i / 64].data[j + k / 4 * 2] = e2m1x8;
        } // end loop k
      } // end loop j

      uint32_t sf0 = sfu32[0];
      uint32_t sf1 = __shfl_xor_sync(0xffffffff, sfu32[1], 1);
      uint32_t sf_final = lane_id % 4 == 0 ? sf0 : sf1;
      sf[n0][8 * i / 64] = sf_final;
    } // end loop i
  } // end loop n0

  return;
}

template <CUtensorMapSwizzle Swizzle>
__launch_bounds__(CONSUMER_THREAD_NUM + PRODUCER_THREAD_NUM, 1) __global__
    void nvfp4_mha_fwd_sm120_kernel(
        BF16 *O, int64_t o_b_stride, int64_t o_h_stride, int64_t o_n_stride,
        int64_t o_d_stride, const float *q_global_scale,
        const float *k_global_scale, const float *v_global_scale, int64_t B,
        int64_t H, int64_t Nq, int64_t Nkv, int64_t Dqk, int64_t Dv,
        float qk_norm, const __grid_constant__ CUtensorMap q_tensor_map,
        const __grid_constant__ CUtensorMap k_tensor_map,
        const __grid_constant__ CUtensorMap v_tensor_map,
        const __grid_constant__ CUtensorMap q_sf_tensor_map,
        const __grid_constant__ CUtensorMap k_sf_tensor_map,
        const __grid_constant__ CUtensorMap v_sf_tensor_map) {
  int block_b = blockIdx.z;                    // batch
  int block_h = blockIdx.y;                    // head
  int block_n = blockIdx.x * TILE_DOT_BLOCK_M; // seq

  int tid = threadIdx.x;
  int lane_id = tid % 32;
  int warp_id = tid / 32;
  bool is_consumer = tid < CONSUMER_THREAD_NUM;
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
  int warp_dot0_m = warp_id * TILE_DOT0_WARP_M;
  int warp_dot0_n = 0;
  int warp_dot1_m = warp_id * TILE_DOT1_WARP_M;
  int warp_dot1_n = 0;

  //===--------------------------------------------------------------------===//
  // Define shared memory for q, k and v
  //===--------------------------------------------------------------------===//

  extern __shared__ __align__(1024) NVFP4x2 smem[];
  NVFP4x2 *q_shared = smem;
  NVFP4x2 *k_shared = q_shared + TILE_Q_BLOCK_ELES / ELES_PER_NVFP4x2;
  NVFP4x2 *v_shared = k_shared + TILE_K_BLOCK_ELES / ELES_PER_NVFP4x2 * STAGE;
  F8E4M3 *sf_smem = reinterpret_cast<F8E4M3 *>(
      v_shared + TILE_V_BLOCK_ELES / ELES_PER_NVFP4x2 * STAGE);
  F8E4M3 *q_sf_shared = sf_smem;
  F8E4M3 *k_sf_shared = q_sf_shared + TILE_Q_SF_BLOCK_ELES * STAGE;
  F8E4M3 *v_sf_shared = k_sf_shared + TILE_K_SF_BLOCK_ELES * STAGE;
  auto get_q_stage_mem = [&](int stage) {
    (void)stage;
    return q_shared;
  };
  auto get_k_stage_mem = [&](int stage) {
    return k_shared + TILE_K_BLOCK_ELES / ELES_PER_NVFP4x2 * stage;
  };
  auto get_v_stage_mem = [&](int stage) {
    return v_shared + TILE_V_BLOCK_ELES / ELES_PER_NVFP4x2 * stage;
  };
  auto get_q_sf_stage_mem = [&](int stage) {
    return q_sf_shared + TILE_Q_SF_BLOCK_ELES * stage;
  };
  auto get_k_sf_stage_mem = [&](int stage) {
    return k_sf_shared + TILE_K_SF_BLOCK_ELES * stage;
  };
  auto get_v_sf_stage_mem = [&](int stage) {
    return v_sf_shared + TILE_V_SF_BLOCK_ELES * stage;
  };

  //===--------------------------------------------------------------------===//
  // Define brrier for a and b tma
  //===--------------------------------------------------------------------===//

  __shared__ __align__(8) uint64_t empty_bar[STAGE];
  __shared__ __align__(8) uint64_t full_bar[STAGE];
  if (is_producer && lane_id == 0) {
#pragma unroll
    for (int i = 0; i < STAGE; ++i) {
      init_barrier(&empty_bar[i], WARP_NUM);
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
    if (tid == CONSUMER_THREAD_NUM) {
      int stage = 0;
      int phase = 0;

      /// Leading TMA Q/K/V and Q_SF/K_SF/V_SF

      /// Acquire
      wait(&empty_bar[stage], phase);
      load_4d_async(get_q_stage_mem(stage), &q_tensor_map, &full_bar[stage],
                    block_b, block_h, block_dot_m, 0);
      load_4d_async(get_q_sf_stage_mem(stage), &q_sf_tensor_map,
                    &full_bar[stage], block_b, block_h, 0, block_dot_m);
      load_4d_async(get_k_stage_mem(stage), &k_tensor_map, &full_bar[stage],
                    block_b, block_h, 0, 0);
      load_4d_async(get_k_sf_stage_mem(stage), &k_sf_tensor_map,
                    &full_bar[stage], block_b, block_h, 0, 0);
      load_4d_async(get_v_stage_mem(stage), &v_tensor_map, &full_bar[stage],
                    block_b, block_h, 0, 0);
      load_4d_async(get_v_sf_stage_mem(stage), &v_sf_tensor_map,
                    &full_bar[stage], block_b, block_h, 0, 0);
      /// Commit
      expect_bytes(&full_bar[stage],
                   TILE_Q_BLOCK_SIZE + TILE_K_BLOCK_SIZE + TILE_V_BLOCK_SIZE +
                       TILE_Q_SF_BLOCK_SIZE + TILE_K_SF_BLOCK_SIZE +
                       TILE_V_SF_BLOCK_SIZE);

      ++stage;

      /// Next
      for (int dot0_n = TILE_DOT0_BLOCK_N; dot0_n < DOT0_N;
           dot0_n += TILE_DOT0_BLOCK_N, ++stage) {
        if (stage == STAGE) {
          stage = 0;
          phase ^= 1;
        }

        /// Acquire
        wait(&empty_bar[stage], phase);

        /// TMA K/V and K_SF/V_SF
        load_4d_async(get_k_stage_mem(stage), &k_tensor_map, &full_bar[stage],
                      block_b, block_h, dot0_n, 0);
        load_4d_async(get_k_sf_stage_mem(stage), &k_sf_tensor_map,
                      &full_bar[stage], block_b, block_h, 0, dot0_n);
        load_4d_async(get_v_stage_mem(stage), &v_tensor_map, &full_bar[stage],
                      block_b, block_h, 0, dot0_n / ELES_PER_NVFP4x2);
        load_4d_async(get_v_sf_stage_mem(stage), &v_sf_tensor_map,
                      &full_bar[stage], block_b, block_h,
                      dot0_n / (BLOCK_SIZE * 4), 0);

        /// Commit
        expect_bytes(&full_bar[stage], TILE_K_BLOCK_SIZE + TILE_V_BLOCK_SIZE +
                                           TILE_K_SF_BLOCK_SIZE +
                                           TILE_V_SF_BLOCK_SIZE);
      }
    }
  } // end if (is_producer)

  //===--------------------------------------------------------------------===//
  // Consumer
  //===--------------------------------------------------------------------===//

  else {
    reg_alloc<200>();
    if (lane_id == 0) {
#pragma unroll
      for (int i = 0; i < STAGE; ++i) {
        arrive(&empty_bar[i], 1);
      }
    }

    //===------------------------------------------------------------------===//
    // Define fragment
    //===------------------------------------------------------------------===//

    /// Q
    AFrag_NVFP4_16x64 q_frag[TILE_DOT0_WARP_M / TILE_ATOMIC_M]
                            [TILE_DOT0_WARP_K / TILE_ATOMIC_K];
    uint32_t q_sf_frag[TILE_DOT0_WARP_M / TILE_ATOMIC_M]
                      [TILE_DOT0_WARP_K / TILE_ATOMIC_K];
    /// max(row max of S=Q @ K)
    float max0[TILE_DOT0_WARP_M / TILE_ATOMIC_M];
    float max1[TILE_DOT0_WARP_M / TILE_ATOMIC_M];
    /// l（sum of exp)
    float l0[TILE_DOT0_WARP_M / TILE_ATOMIC_M];
    float l1[TILE_DOT0_WARP_M / TILE_ATOMIC_M];
#pragma unroll
    for (int i = 0; i < TILE_DOT0_WARP_M / TILE_ATOMIC_M; ++i) {
      max0[i] = -INFINITY;
      max1[i] = -INFINITY;
      l0[i] = 0.0f;
      l1[i] = 0.0f;
    }
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

    for (int dot0_n = 0, stage = 0, phase = 0; dot0_n < DOT0_N;
         dot0_n += TILE_DOT0_BLOCK_N, ++stage) {
      if (stage == STAGE) {
        stage = 0;
        phase ^= 1;
      }

      block_dot0_n = dot0_n;

      /// Wait
      wait(&full_bar[stage], phase);

      if (dot0_n == 0) {
        /// Load shared to fragment for Q/Q_SF operand

#pragma unroll
        for (int i = 0; i < TILE_DOT0_WARP_M / TILE_ATOMIC_M; ++i) {
#pragma unroll
          for (int j = 0; j < TILE_DOT0_WARP_K / TILE_ATOMIC_K; ++j) {
            /// Q
            int *q_regs = (int *)q_frag[i][j].data;
            int q_shared_row = warp_dot0_m + i * TILE_ATOMIC_M + (lane_id % 16);
            int q_shared_col =
                (j * TILE_ATOMIC_K + (lane_id / 16) * 32) / ELES_PER_NVFP4x2;
            int q_shared_row_swiz = SwizzleIndexMap<Swizzle>::get_row(
                q_shared_row, q_shared_col, sizeof(NVFP4x2));
            int q_shared_col_swiz = SwizzleIndexMap<Swizzle>::get_col(
                q_shared_row, q_shared_col, sizeof(NVFP4x2));
            uint32_t q_addr = __cvta_generic_to_shared(
                get_q_stage_mem(stage) +
                q_shared_row_swiz * TILE_DOT0_BLOCK_K / ELES_PER_NVFP4x2 +
                q_shared_col_swiz);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                         "{%0, %1, %2, %3}, [%4];"
                         : "=r"(q_regs[0]), "=r"(q_regs[1]), "=r"(q_regs[2]),
                           "=r"(q_regs[3])
                         : "r"(q_addr));

            /// Q_SF
            if (lane_id % 4 < 2) {
              // m map to non-quantize
              int m = warp_dot0_m + i * TILE_ATOMIC_M + (lane_id % 2) * 8 +
                      lane_id / 4;
              // k map to quantize
              int k = j * TILE_ATOMIC_K;
              const auto *q_sf_tile =
                  get_q_sf_stage_mem(stage) +
                  (k / (BLOCK_SIZE * 4)) * TILE_DOT0_BLOCK_M * 4 + m * 4;
              q_sf_frag[i][j] = *reinterpret_cast<const uint32_t *>(q_sf_tile);
            }
          } // end loop i
        } // end loop j
      } // end if (dot0_n == 0) leading

      /// S
      DFrag_F32_16x8 s_frag[TILE_DOT0_WARP_M / TILE_ATOMIC_M]
                           [TILE_DOT0_WARP_N / TILE_ATOMIC_N];

      /// Step 0: S = Q @ K

#pragma unroll
      for (int dot0_k = 0; dot0_k < TILE_DOT0_BLOCK_K;
           dot0_k += TILE_ATOMIC_K) {
        int dot0_atomic_k_cnt = dot0_k / TILE_ATOMIC_K;
#pragma unroll
        for (int dot0_atomic_m = 0; dot0_atomic_m < TILE_DOT0_WARP_M;
             dot0_atomic_m += TILE_ATOMIC_M) {
          int dot0_atomic_m_cnt = dot0_atomic_m / TILE_ATOMIC_M;
#pragma unroll
          for (int dot0_atomic_n = 0; dot0_atomic_n < TILE_DOT0_WARP_N;
               dot0_atomic_n += TILE_ATOMIC_N) {
            int dot0_atomic_n_cnt = dot0_atomic_n / TILE_ATOMIC_N;
            /// Load K
            BFrag_NVFP4_64x8 k_frag;
            {
              int *k_regs = (int *)k_frag.data;
              int k_shared_row = warp_dot0_n + dot0_atomic_n + (lane_id % 8);
              int k_shared_col =
                  (dot0_k + ((lane_id % 16) / 8) * 32) / ELES_PER_NVFP4x2;
              int k_shared_row_swiz = SwizzleIndexMap<Swizzle>::get_row(
                  k_shared_row, k_shared_col, sizeof(NVFP4x2));
              int k_shared_col_swiz = SwizzleIndexMap<Swizzle>::get_col(
                  k_shared_row, k_shared_col, sizeof(NVFP4x2));
              uint32_t k_addr = __cvta_generic_to_shared(
                  get_k_stage_mem(stage) +
                  k_shared_row_swiz * TILE_DOT0_BLOCK_K / ELES_PER_NVFP4x2 +
                  k_shared_col_swiz);
              asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                           "{%0, %1}, [%2];"
                           : "=r"(k_regs[0]), "=r"(k_regs[1])
                           : "r"(k_addr));
            }
            /// Load K_SF
            uint32_t k_sf_frag = 0;
            if (lane_id % 4 == 0) {
              // map k to quantize
              int k = dot0_k;
              // map n to non-quantize
              int n = warp_dot0_n + dot0_atomic_n + (lane_id / 4);
              const auto *k_sf_tile =
                  get_k_sf_stage_mem(stage) +
                  (k / (BLOCK_SIZE * 4)) * TILE_DOT0_BLOCK_N * 4 + n * 4;
              k_sf_frag = *reinterpret_cast<const uint32_t *>(k_sf_tile);
            }

            /// Dot0
            if (dot0_k == 0) {
              fma(s_frag[dot0_atomic_m_cnt][dot0_atomic_n_cnt].data[0],
                  s_frag[dot0_atomic_m_cnt][dot0_atomic_n_cnt].data[1],
                  s_frag[dot0_atomic_m_cnt][dot0_atomic_n_cnt].data[2],
                  s_frag[dot0_atomic_m_cnt][dot0_atomic_n_cnt].data[3],
                  q_frag[dot0_atomic_m_cnt][dot0_atomic_k_cnt].data[0],
                  q_frag[dot0_atomic_m_cnt][dot0_atomic_k_cnt].data[1],
                  q_frag[dot0_atomic_m_cnt][dot0_atomic_k_cnt].data[2],
                  q_frag[dot0_atomic_m_cnt][dot0_atomic_k_cnt].data[3],
                  k_frag.data[0], k_frag.data[1], 0, 0, 0, 0,
                  q_sf_frag[dot0_atomic_m_cnt][dot0_atomic_k_cnt], k_sf_frag);
            } else {
              fma(s_frag[dot0_atomic_m_cnt][dot0_atomic_n_cnt].data[0],
                  s_frag[dot0_atomic_m_cnt][dot0_atomic_n_cnt].data[1],
                  s_frag[dot0_atomic_m_cnt][dot0_atomic_n_cnt].data[2],
                  s_frag[dot0_atomic_m_cnt][dot0_atomic_n_cnt].data[3],
                  q_frag[dot0_atomic_m_cnt][dot0_atomic_k_cnt].data[0],
                  q_frag[dot0_atomic_m_cnt][dot0_atomic_k_cnt].data[1],
                  q_frag[dot0_atomic_m_cnt][dot0_atomic_k_cnt].data[2],
                  q_frag[dot0_atomic_m_cnt][dot0_atomic_k_cnt].data[3],
                  k_frag.data[0], k_frag.data[1],
                  s_frag[dot0_atomic_m_cnt][dot0_atomic_n_cnt].data[0],
                  s_frag[dot0_atomic_m_cnt][dot0_atomic_n_cnt].data[1],
                  s_frag[dot0_atomic_m_cnt][dot0_atomic_n_cnt].data[2],
                  s_frag[dot0_atomic_m_cnt][dot0_atomic_n_cnt].data[3],
                  q_sf_frag[dot0_atomic_m_cnt][dot0_atomic_k_cnt], k_sf_frag);
            }
          } // end loop dot0_atomic_n
        } // end loop dot0_atomic_m
      } // end loop dot0_k

      /// Step 1: S = S * alpha0(nvfp4 global scale) * qk_norm(rsqrt(dk))
      float qk_scale = (*q_global_scale) * (*k_global_scale) * qk_norm;
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

      /// Step 2: max = rowmax(S)
      float max_new0[TILE_DOT0_WARP_M / TILE_ATOMIC_M];
      float max_new1[TILE_DOT0_WARP_M / TILE_ATOMIC_M];
#pragma unroll
      for (int i = 0; i < TILE_DOT0_WARP_M / TILE_ATOMIC_M; ++i) {
        max_new0[i] = max0[i];
        max_new1[i] = max1[i];
#pragma unroll
        for (int j = 0; j < TILE_DOT0_WARP_N / TILE_ATOMIC_N; ++j) {
          max_new0[i] = fmaxf(max_new0[i], s_frag[i][j].data[0]);
          max_new0[i] = fmaxf(max_new0[i], s_frag[i][j].data[1]);
          max_new1[i] = fmaxf(max_new1[i], s_frag[i][j].data[2]);
          max_new1[i] = fmaxf(max_new1[i], s_frag[i][j].data[3]);
        } // end loop j
        max_new0[i] =
            fmaxf(__shfl_xor_sync(0xffffffff, max_new0[i], 1), max_new0[i]);
        max_new0[i] =
            fmaxf(__shfl_xor_sync(0xffffffff, max_new0[i], 2), max_new0[i]);
        max_new1[i] =
            fmaxf(__shfl_xor_sync(0xffffffff, max_new1[i], 1), max_new1[i]);
        max_new1[i] =
            fmaxf(__shfl_xor_sync(0xffffffff, max_new1[i], 2), max_new1[i]);
      } // end loop i

      /// Step 3: p = exp(S - max)
      DFrag_F32_16x8 p_frag[TILE_DOT0_WARP_M / TILE_ATOMIC_M]
                           [TILE_DOT0_WARP_N / TILE_ATOMIC_N];
#pragma unroll
      for (int i = 0; i < TILE_DOT0_WARP_M / TILE_ATOMIC_M; ++i) {
#pragma unroll
        for (int j = 0; j < TILE_DOT0_WARP_N / TILE_ATOMIC_N; ++j) {
          p_frag[i][j].data[0] = fast_exp(s_frag[i][j].data[0], max_new0[i]);
          p_frag[i][j].data[1] = fast_exp(s_frag[i][j].data[1], max_new0[i]);
          p_frag[i][j].data[2] = fast_exp(s_frag[i][j].data[2], max_new1[i]);
          p_frag[i][j].data[3] = fast_exp(s_frag[i][j].data[3], max_new1[i]);
        } // end loop j
      } // end loop i

      /// Step 4: l = sum(p)
      float l_new0[TILE_DOT0_WARP_M / TILE_ATOMIC_M];
      float l_new1[TILE_DOT0_WARP_M / TILE_ATOMIC_M];
#pragma unroll
      for (int i = 0; i < TILE_DOT0_WARP_M / TILE_ATOMIC_M; ++i) {
        l_new0[i] = p_frag[i][0].data[0] + p_frag[i][0].data[1];
        l_new1[i] = p_frag[i][0].data[2] + p_frag[i][0].data[3];
#pragma unroll
        for (int j = 1; j < TILE_DOT0_WARP_N / TILE_ATOMIC_N; ++j) {
          l_new0[i] += p_frag[i][j].data[0];
          l_new0[i] += p_frag[i][j].data[1];
          l_new1[i] += p_frag[i][j].data[2];
          l_new1[i] += p_frag[i][j].data[3];
        } // end loop j
        l_new0[i] += __shfl_xor_sync(0xffffffff, l_new0[i], 1);
        l_new0[i] += __shfl_xor_sync(0xffffffff, l_new0[i], 2);
        l_new1[i] += __shfl_xor_sync(0xffffffff, l_new1[i], 1);
        l_new1[i] += __shfl_xor_sync(0xffffffff, l_new1[i], 2);

        /// Fix exp sum
        /// redundant p is exp(0 - max)
        /// total redundant exp sum is exp(-max) * (align_up(N,
        /// TILE_DOT0_BLOCK_N) - N)
        if (dot0_n + TILE_DOT0_BLOCK_N > DOT0_N) {
          l_new0[i] -= (align_up(DOT0_N, TILE_DOT0_BLOCK_N) - DOT0_N) /
                       fast_exp(max_new0[i]);
          l_new1[i] -= (align_up(DOT0_N, TILE_DOT0_BLOCK_N) - DOT0_N) /
                       fast_exp(max_new1[i]);
        }

        // Update l
        l0[i] = fast_exp(max0[i], max_new0[i]) * l0[i] + l_new0[i];
        l1[i] = fast_exp(max1[i], max_new1[i]) * l1[i] + l_new1[i];
      } // end loop i

      /// Step 4: o = expf(max - max_new) * o
      ///         4 * 8 + 2 = 34

#pragma unroll
      for (int i = 0; i < TILE_DOT1_WARP_M / TILE_ATOMIC_M; ++i) {
#pragma unroll
        for (int j = 0; j < TILE_DOT1_WARP_N / TILE_ATOMIC_N; ++j) {
          o_frag[i][j].data[0] *= fast_exp(max0[i], max_new0[i]);
          o_frag[i][j].data[1] *= fast_exp(max0[i], max_new0[i]);
          o_frag[i][j].data[2] *= fast_exp(max1[i], max_new1[i]);
          o_frag[i][j].data[3] *= fast_exp(max1[i], max_new1[i]);
        }
      }
/// Update max
#pragma unroll
      for (int i = 0; i < TILE_DOT1_WARP_M / TILE_ATOMIC_M; ++i) {
        max0[i] = max_new0[i];
        max1[i] = max_new1[i];
      }

      /// Step 5: quantize P from f32 to nvfp4
      AFrag_NVFP4_16x64 p_fp4_frag[TILE_DOT1_WARP_M / TILE_ATOMIC_M]
                                  [TILE_DOT1_WARP_K / TILE_ATOMIC_K];
      uint32_t p_fp4_sf_frag[TILE_DOT1_WARP_M / TILE_ATOMIC_M]
                            [TILE_DOT1_WARP_K / TILE_ATOMIC_K];
      convert_to_nvfp4(p_frag, p_fp4_frag, p_fp4_sf_frag);

      /// Step 6: o = P @ V
      int dot1_k = dot0_n;

#pragma unroll
      for (int dot1_atomic_k = 0; dot1_atomic_k < TILE_DOT1_BLOCK_K;
           dot1_atomic_k += TILE_ATOMIC_K) {
        int dot1_atomic_k_cnt = dot1_atomic_k / TILE_ATOMIC_K;
#pragma unroll
        for (int dot1_atomic_m = 0; dot1_atomic_m < TILE_DOT1_WARP_M;
             dot1_atomic_m += TILE_ATOMIC_M) {
          int dot1_atomic_m_cnt = dot1_atomic_m / TILE_ATOMIC_M;
#pragma unroll
          for (int dot1_atomic_n = 0; dot1_atomic_n < TILE_DOT1_WARP_N;
               dot1_atomic_n += TILE_ATOMIC_N) {
            int dot1_atomic_n_cnt = dot1_atomic_n / TILE_ATOMIC_N;
            /// Load V
            BFrag_NVFP4_64x8 v_frag;
            {
              int *v_regs = (int *)v_frag.data;
              int v_shared_row = warp_dot1_n + dot1_atomic_n + (lane_id % 8);
              int v_shared_col = (dot1_atomic_k + ((lane_id % 16) / 8) * 32) /
                                 ELES_PER_NVFP4x2;
              int v_shared_row_swiz = SwizzleIndexMap<Swizzle>::get_row(
                  v_shared_row, v_shared_col, sizeof(NVFP4x2));
              int v_shared_col_swiz = SwizzleIndexMap<Swizzle>::get_col(
                  v_shared_row, v_shared_col, sizeof(NVFP4x2));
              uint32_t v_addr = __cvta_generic_to_shared(
                  get_v_stage_mem(stage) +
                  v_shared_row_swiz * TILE_DOT1_BLOCK_K / ELES_PER_NVFP4x2 +
                  v_shared_col_swiz);
              asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                           "{%0, %1}, [%2];"
                           : "=r"(v_regs[0]), "=r"(v_regs[1])
                           : "r"(v_addr));
            }
            /// Load V_SF
            uint32_t v_sf_frag = 0;
            if (lane_id % 4 == 0) {
              // map k to quantize
              int k = dot1_atomic_k;
              // map n to non-quantize
              int n = warp_dot1_n + dot1_atomic_n + (lane_id / 4);
              const auto *v_sf_tile =
                  get_v_sf_stage_mem(stage) +
                  (k / (BLOCK_SIZE * 4)) * TILE_DOT1_BLOCK_N * 4 + n * 4;
              v_sf_frag = *reinterpret_cast<const uint32_t *>(v_sf_tile);
            }

            /// Dot1
            fma(o_frag[dot1_atomic_m_cnt][dot1_atomic_n_cnt].data[0],
                o_frag[dot1_atomic_m_cnt][dot1_atomic_n_cnt].data[1],
                o_frag[dot1_atomic_m_cnt][dot1_atomic_n_cnt].data[2],
                o_frag[dot1_atomic_m_cnt][dot1_atomic_n_cnt].data[3],
                p_fp4_frag[dot1_atomic_m_cnt][dot1_atomic_k_cnt].data[0],
                p_fp4_frag[dot1_atomic_m_cnt][dot1_atomic_k_cnt].data[1],
                p_fp4_frag[dot1_atomic_m_cnt][dot1_atomic_k_cnt].data[2],
                p_fp4_frag[dot1_atomic_m_cnt][dot1_atomic_k_cnt].data[3],
                v_frag.data[0], v_frag.data[1],
                o_frag[dot1_atomic_m_cnt][dot1_atomic_n_cnt].data[0],
                o_frag[dot1_atomic_m_cnt][dot1_atomic_n_cnt].data[1],
                o_frag[dot1_atomic_m_cnt][dot1_atomic_n_cnt].data[2],
                o_frag[dot1_atomic_m_cnt][dot1_atomic_n_cnt].data[3],
                p_fp4_sf_frag[dot1_atomic_m_cnt][dot1_atomic_k_cnt], v_sf_frag);
          } // end loop dot1_atomic_n
        } // end loop dot1_atomic_m
      } // end loop dot1_atomic_k

      /// Release
      if (lane_id == 0) {
        arrive(&empty_bar[stage], 1);
      }
    } // end loop dot0_n/dot1_k

    //===------------------------------------------------------------------===//
    // Epilogue
    //===------------------------------------------------------------------===//

    /// step 8: o = o / l * alpha1(V global scale) * p_scale(1/(448 * 6))

#pragma unroll
    for (int i = 0; i < TILE_DOT1_WARP_M / TILE_ATOMIC_M; ++i) {
#pragma unroll
      for (int j = 0; j < TILE_DOT1_WARP_N / TILE_ATOMIC_N; ++j) {
        float scale0 = (1.0f / l0[i]) * (1.0f / (448 * 6)) * *v_global_scale;
        float scale1 = (1.0f / l1[i]) * (1.0f / (448 * 6)) * *v_global_scale;
        o_frag[i][j].data[0] *= scale0;
        o_frag[i][j].data[1] *= scale0;
        o_frag[i][j].data[2] *= scale1;
        o_frag[i][j].data[3] *= scale1;
      }
    }

    /// Step 9: Write back to O

    if (block_dot_m + warp_dot0_m + TILE_DOT0_WARP_M <= Nq &&
        block_dot_n + warp_dot0_n + TILE_DOT1_WARP_N <= Dv) {
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
    } else {
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
// ac4k mha attention with nvfp4 acceleration
// Q               :[B,   H,    Nq,     Dqk/2]xNVFP4
// K               :[B,   H,    Nkv,    Dqk/2]xNVFP4
// V               :[B,   H,    Dv,     Nkv/2]xNVFP4
// O               :[B,   H,    Nq,     Dv]xNVFP4
// Q_SF            :[B,   H,    Dqk/64, Nq,   4]xF8
// K_SF            :[B,   H,    Dqk/64, Nkv,  4]xF8
// V_SF            :[B,   H,    Dv/64,  Nkv,  4]xF8
// Q_GLOBAL_SCALE  :F32
// K_GLOBAL_SCALE  :F32
// V_GLOBAL_SCALE  :F32
// Dqk             : Head dim size for Q and K
//===--------------------------------------------------------------------===//

void nvfp4_mha_fwd_sm120(torch::Tensor &o, torch::Tensor &q,
                         torch::Tensor &q_sf, torch::Tensor &q_global_scale,
                         torch::Tensor &k, torch::Tensor &k_sf,
                         torch::Tensor &k_global_scale, torch::Tensor &v,
                         torch::Tensor &v_sf, torch::Tensor &v_global_scale,
                         int64_t Dqk) {
  /// CHECK Q & Q_SF
  CHECK_INPUT(q, at::ScalarType::Byte, "Q must be pack to uint8 tensor");
  TORCH_CHECK(q.dim() == 4, "Q must be a 4D tensor");
  int64_t B = q.size(0);
  int64_t H = q.size(1);
  int64_t Nq = q.size(2);
  TORCH_CHECK(q.size(3) == align_up(Dqk, BLOCK_SIZE * 4) / ELES_PER_NVFP4x2,
              "q.size(3) must be ",
              align_up(Dqk, BLOCK_SIZE * 4) / ELES_PER_NVFP4x2, " but see ",
              q.size(3));
  /// TODO(need remove limit)
  TORCH_CHECK(Dqk <= HEAD_DIM_ALIGN_SIZE, "Dqk must be less than ",
              HEAD_DIM_ALIGN_SIZE);
  CHECK_INPUT(q_sf, at::ScalarType::Float8_e4m3fn,
              "q_sf must be a f8e4m3 tensor");
  TORCH_CHECK(q_sf.dim() == 5, "Q_SF must be a 5D tensor");
  TORCH_CHECK(q_sf.size(0) == B, "Meet invalid q_sf size(0) with ",
              q_sf.size(0), "==", B);
  TORCH_CHECK(q_sf.size(1) == H, "Meet invalid q_sf size(1) with ",
              q_sf.size(1), "==", H);
  TORCH_CHECK(q_sf.size(2) == ceil_div(Dqk, BLOCK_SIZE * 4),
              "Meet invalid q_sf size(2) with ", q_sf.size(2),
              "==", ceil_div(Dqk, BLOCK_SIZE * 4));
  TORCH_CHECK(q_sf.size(3) == align_up(Nq, CROSS_DIM_ALIGN_SIZE),
              "Meet invalid q_sf size(3) with ", q_sf.size(3),
              "==", align_up(Nq, CROSS_DIM_ALIGN_SIZE));
  TORCH_CHECK(q_sf.size(4) == 4, "Meet invalid q_sf size(4) with ",
              q_sf.size(4), "==", 4);
  CHECK_SCALAR(q_global_scale, at::ScalarType::Float,
               "q_global_scale must be a float tensor");

  /// CHECK K & K_SF
  CHECK_INPUT(k, at::ScalarType::Byte, "K must be pack to uint8 tensor");
  TORCH_CHECK(k.dim() == 4, "K must be a 4D tensor");
  TORCH_CHECK(k.size(0) == B, "K must have the same batch size as Q");
  TORCH_CHECK(k.size(1) == H, "K must have the same head number as Q");
  int64_t Nkv = k.size(2);
  TORCH_CHECK(k.size(3) == q.size(3), "K must have the same dim as Q");
  CHECK_INPUT(k_sf, at::ScalarType::Float8_e4m3fn,
              "K_SF must be a f8e4m3 tensor");
  TORCH_CHECK(k_sf.dim() == 5, "k_sf must be a 5D tensor");
  TORCH_CHECK(k_sf.size(0) == B, "Meet invalid k_sf size(0) with ",
              k_sf.size(0), "==", B);
  TORCH_CHECK(k_sf.size(1) == H, "Meet invalid k_sf size(1) with ",
              k_sf.size(1), "==", H);
  TORCH_CHECK(k_sf.size(2) == ceil_div(Dqk, BLOCK_SIZE * 4),
              "Meet invalid k_sf size(2) with ", k_sf.size(2),
              "==", ceil_div(Dqk, BLOCK_SIZE * 4));
  TORCH_CHECK(k_sf.size(3) == align_up(Nkv, CROSS_DIM_ALIGN_SIZE),
              "Meet invalid k_sf size(3) with ", k_sf.size(3),
              "==", align_up(Nkv, CROSS_DIM_ALIGN_SIZE));
  TORCH_CHECK(k_sf.size(4) == 4, "Meet invalid k_sf size(4) with ",
              k_sf.size(4), "==", 4);
  CHECK_SCALAR(k_global_scale, at::ScalarType::Float,
               "k_global_scale must be a float tensor");

  /// CHECK V & V_SF
  CHECK_INPUT(v, at::ScalarType::Byte, "V must be pack to uint8 tensor");
  TORCH_CHECK(v.dim() == 4, "V must be a 4D tensor");
  TORCH_CHECK(v.size(0) == B, "V must have the same batch size as Q");
  TORCH_CHECK(v.size(1) == H, "V must have the same head number as Q");
  int64_t Dv = v.size(2);
  TORCH_CHECK(v.size(3) * ELES_PER_NVFP4x2 == align_up(Nkv, BLOCK_SIZE * 4),
              "V must have the same sequence length as K");
  /// TODO(need remove limit)
  TORCH_CHECK(Dv <= HEAD_DIM_ALIGN_SIZE, "Dqk must be less than ",
              HEAD_DIM_ALIGN_SIZE);
  CHECK_INPUT(v_sf, at::ScalarType::Float8_e4m3fn,
              "V_SF must be a f8e4m3 tensor");
  TORCH_CHECK(v_sf.dim() == 5, "V_SF must be a 5D tensor");
  TORCH_CHECK(v_sf.size(0) == B, "Meet invalid v_sf size(0) with ",
              v_sf.size(0), "==", B);
  TORCH_CHECK(v_sf.size(1) == H, "Meet invalid v_sf size(1) with ",
              v_sf.size(1), "==", H);
  TORCH_CHECK(v_sf.size(2) == ceil_div(Nkv, BLOCK_SIZE * 4),
              "Meet invalid v_sf size(2) with ", v_sf.size(2),
              "==", ceil_div(Nkv, BLOCK_SIZE * 4));
  TORCH_CHECK(v_sf.size(3) == align_up(Dv, CROSS_DIM_ALIGN_SIZE),
              "Meet invalid v_sf size(3) with ", v_sf.size(3),
              "==", align_up(Dv, CROSS_DIM_ALIGN_SIZE));
  TORCH_CHECK(v_sf.size(4) == 4, "Meet invalid v_sf size(4) with ",
              v_sf.size(4), "==", 4);
  CHECK_SCALAR(v_global_scale, at::ScalarType::Float,
               "v_global_scale must be a float tensor");

  /// CHECK O
  CHECK_OUTPUT(o, at::ScalarType::BFloat16, "O must be a bfloat16 tensor");
  TORCH_CHECK(o.dim() == 4, "O must be a 4D tensor");
  TORCH_CHECK(o.size(0) == B, "O must have the same batch size as Q");
  TORCH_CHECK(o.size(1) == H, "O must have the same head number as Q");
  TORCH_CHECK(o.size(2) == Nq, "O must have the same sequence length as Q");
  TORCH_CHECK(o.size(3) == Dv, "O must have the same dim as V");

  /// TMA descriptor
  const auto SWIZZLE = CU_TENSOR_MAP_SWIZZLE_64B;
  CUtensorMap q_tensor_map = create_4d_tensor_map<
      1, 1, TILE_DOT0_BLOCK_M, TILE_DOT0_BLOCK_K / ELES_PER_NVFP4x2,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8, NVFP4x2, SWIZZLE>(
      reinterpret_cast<const NVFP4x2 *>(q.data_ptr()),
      static_cast<uint64_t>(q.size(0)), static_cast<uint64_t>(q.size(1)),
      static_cast<uint64_t>(q.size(2)), static_cast<uint64_t>(q.size(3)));
  CUtensorMap k_tensor_map = create_4d_tensor_map<
      1, 1, TILE_DOT0_BLOCK_N, TILE_DOT0_BLOCK_K / ELES_PER_NVFP4x2,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8, NVFP4x2, SWIZZLE>(
      reinterpret_cast<const NVFP4x2 *>(k.data_ptr()),
      static_cast<uint64_t>(k.size(0)), static_cast<uint64_t>(k.size(1)),
      static_cast<uint64_t>(k.size(2)), static_cast<uint64_t>(k.size(3)));
  CUtensorMap v_tensor_map = create_4d_tensor_map<
      1, 1, TILE_DOT1_BLOCK_N, TILE_DOT1_BLOCK_K / ELES_PER_NVFP4x2,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8, NVFP4x2, SWIZZLE>(
      reinterpret_cast<const NVFP4x2 *>(v.data_ptr()),
      static_cast<uint64_t>(v.size(0)), static_cast<uint64_t>(v.size(1)),
      static_cast<uint64_t>(v.size(2)), static_cast<uint64_t>(v.size(3)));

  CUtensorMap q_sf_tensor_map = create_4d_tensor_map<
      1, 1, TILE_DOT0_BLOCK_K / (BLOCK_SIZE * 4), TILE_DOT0_BLOCK_M,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32, uint32_t>(
      reinterpret_cast<const uint32_t *>(q_sf.data_ptr()),
      static_cast<uint64_t>(q_sf.size(0)), static_cast<uint64_t>(q_sf.size(1)),
      static_cast<uint64_t>(q_sf.size(2)), static_cast<uint64_t>(q_sf.size(3)));
  CUtensorMap k_sf_tensor_map = create_4d_tensor_map<
      1, 1, TILE_DOT0_BLOCK_K / (BLOCK_SIZE * 4), TILE_DOT0_BLOCK_N,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32, uint32_t>(
      reinterpret_cast<const uint32_t *>(k_sf.data_ptr()),
      static_cast<uint64_t>(k_sf.size(0)), static_cast<uint64_t>(k_sf.size(1)),
      static_cast<uint64_t>(k_sf.size(2)), static_cast<uint64_t>(k_sf.size(3)));
  CUtensorMap v_sf_tensor_map = create_4d_tensor_map<
      1, 1, TILE_DOT1_BLOCK_K / (BLOCK_SIZE * 4), TILE_DOT1_BLOCK_N,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32, uint32_t>(
      reinterpret_cast<const uint32_t *>(v_sf.data_ptr()),
      static_cast<uint64_t>(v_sf.size(0)), static_cast<uint64_t>(v_sf.size(1)),
      static_cast<uint64_t>(v_sf.size(2)), static_cast<uint64_t>(v_sf.size(3)));

  /// Set dynamic shared memory size
  auto *kernel = nvfp4_mha_fwd_sm120_kernel<SWIZZLE>;
  size_t smem_size =
      TILE_Q_BLOCK_SIZE +
      (TILE_K_BLOCK_SIZE + TILE_V_BLOCK_SIZE + TILE_Q_SF_BLOCK_SIZE +
       TILE_K_SF_BLOCK_SIZE + TILE_V_SF_BLOCK_SIZE) *
          STAGE;
  CHECK_CUDA_ERROR(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  /// Get CUDA stream
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  /// Launch kernel
  dim3 grid(ceil_div(Nq, static_cast<int64_t>(TILE_DOT_BLOCK_M)), H, B);
  dim3 block(CONSUMER_THREAD_NUM + PRODUCER_THREAD_NUM);
  kernel<<<grid, block, smem_size, stream>>>(
      reinterpret_cast<BF16 *>(o.data_ptr()),
      /* o b stride */ o.stride(0),
      /* o h stride */ o.stride(1),
      /* o n stride */ o.stride(2),
      /* o d stride */ o.stride(3),
      reinterpret_cast<const float *>(q_global_scale.data_ptr()),
      reinterpret_cast<const float *>(k_global_scale.data_ptr()),
      reinterpret_cast<const float *>(v_global_scale.data_ptr()), B, H, Nq, Nkv,
      Dqk, Dv, 1.0f / std::sqrt(static_cast<float>(Dqk)), q_tensor_map,
      k_tensor_map, v_tensor_map, q_sf_tensor_map, k_sf_tensor_map,
      v_sf_tensor_map);
}

} // namespace ac4k
