#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
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

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

using NVFP4x2 = uint8_t;
using E4M3 = uint8_t;
constexpr int ELES_PER_NVFP4x2 = 2;
constexpr int BLOCK_SIZE = 16;

//===----------------------------------------------------------------------===//
// Define tile size for block level, warp level and atomic level
//===----------------------------------------------------------------------===//

const int GROUP_SIZE_M = 16;

/// Block level tile size
const int TILE_BLOCK_M = 128;
const int TILE_BLOCK_N = 128;
const int TILE_BLOCK_K = 128;
const int TILE_BLOCK_PACK_K = TILE_BLOCK_K / ELES_PER_NVFP4x2;
const int TILE_BLOCK_A_ELES = TILE_BLOCK_M * TILE_BLOCK_K;
const int TILE_BLOCK_B_ELES = TILE_BLOCK_N * TILE_BLOCK_K;
const int TILE_IN_ELES = TILE_BLOCK_A_ELES + TILE_BLOCK_B_ELES;
const int TILE_BLOCK_A_SF_ELES = TILE_BLOCK_M * TILE_BLOCK_K / 64;
const int TILE_BLOCK_B_SF_ELES = TILE_BLOCK_N * TILE_BLOCK_K / 64;

/// Warp level tile size
const int TILE_WARP_M = 64;
const int TILE_WARP_N = 64;
const int TILE_WARP_K = 64;
const int TILE_WARP_PACK_K = TILE_WARP_K / ELES_PER_NVFP4x2;

/// Atomic level tile size
const int TILE_ATOMIC_M = 16;
const int TILE_ATOMIC_N = 8;
const int TILE_ATOMIC_K = 64;
const int TILE_ATOMIC_PACK_K = TILE_ATOMIC_K / ELES_PER_NVFP4x2;

const int WARP_M_NUM = TILE_BLOCK_M / TILE_WARP_M;
const int WARP_N_NUM = TILE_BLOCK_N / TILE_WARP_N;
const int WARP_NUM = WARP_M_NUM * WARP_N_NUM;

//===----------------------------------------------------------------------===//
// Define mutiple buffer stage
//===----------------------------------------------------------------------===//

const int STAGE = 2;

//===----------------------------------------------------------------------===//
// Consumer & Producer
//===----------------------------------------------------------------------===//

const int CONSUMER_THREAD_NUM = WARP_NUM * 32;
const int PRODUCER_THREAD_NUM = 32;

//===----------------------------------------------------------------------===//
// TMA
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
    // int index = (row * 64 + col * bpe) / sizeof(int4);
    // // int new_row = index / (128 / sizeof(int4));
    // int new_row = row * 64 / 128;
    // int new_col = index % (128 / sizeof(int4));
    // int row_swiz = new_row;
    // int col_swiz = (row_swiz % 4) ^ new_col;
    // int index_swiz = (row_swiz * 128 + col_swiz * sizeof(int4)) / bpe;
    // return index_swiz % (64 / bpe);

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
// Program ID
//===----------------------------------------------------------------------===//

__device__ static __forceinline__ int2 get_program_id(int M, int N) {
  /// Program ID
  int pid = blockIdx.x;
  /// Number of program ids along the M axis
  int num_pid_m = ceil_div(M, TILE_BLOCK_M);
  /// Number of programs ids along the N axis
  int num_pid_n = ceil_div(N, TILE_BLOCK_N);
  int num_pid_in_group = GROUP_SIZE_M * num_pid_n;
  int group_id = pid / num_pid_in_group;
  int first_pid_m = group_id * GROUP_SIZE_M;
  int group_size_m = (num_pid_m - first_pid_m) < GROUP_SIZE_M
                         ? num_pid_m - first_pid_m
                         : GROUP_SIZE_M;
  int pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m);
  int pid_n = (pid % num_pid_in_group) / group_size_m;

  return make_int2(pid_m, pid_n);
}

//===----------------------------------------------------------------------===//
// Matrix matmul kernel
// D bf16 dtype
// A/B nvfp4 dtype
// accumulator dtype:float
// A shape:MxK stride:Kx1 matrix, row major
// B shape:KxN stride:1xK matrix, col major
// C shape:MxN stride:Nx1 matrix, row major
//===----------------------------------------------------------------------===//

template <at::ScalarType D_AT_TYPE, at::ScalarType BIAS_AT_TYPE,
          CUtensorMapSwizzle Swizzle>
__launch_bounds__(CONSUMER_THREAD_NUM + PRODUCER_THREAD_NUM) __global__
    void nvfp4_dot_scale_sm120_kernel(
        typename AtDataTraits<D_AT_TYPE>::Type *__restrict__ D,
        const float *__restrict__ A_global_scale,
        const float *__restrict__ B_global_scale,
        const typename AtDataTraits<BIAS_AT_TYPE>::Type *__restrict__ bias,
        int M, int N, int K, const __grid_constant__ CUtensorMap a_tensor_map,
        const __grid_constant__ CUtensorMap b_tensor_map,
        const __grid_constant__ CUtensorMap a_sf_tensor_map,
        const __grid_constant__ CUtensorMap b_sf_tensor_map) {
  using D_TYPE = typename AtDataTraits<D_AT_TYPE>::Type;
  using D_TYPEx2 = typename AtDataTraits<D_AT_TYPE>::Typex2;

  //===--------------------------------------------------------------------===//
  // Distribute for block, warp
  //===--------------------------------------------------------------------===//

  int tid = threadIdx.x;
  bool is_consumer = tid < CONSUMER_THREAD_NUM;
  bool is_producer = tid >= CONSUMER_THREAD_NUM;

  int2 pid = get_program_id(M, N);
  int block_m = pid.x * TILE_BLOCK_M;
  int block_n = pid.y * TILE_BLOCK_N;

  int warp_id = tid / 32;
  int warp_id_m = warp_id / WARP_N_NUM;
  int warp_id_n = warp_id % WARP_N_NUM;
  int warp_m = warp_id_m * TILE_WARP_M;
  int warp_n = warp_id_n * TILE_WARP_N;
  int lane_id = tid % 32;

  //===--------------------------------------------------------------------===//
  // Define fragment for a, b and c operand
  //===--------------------------------------------------------------------===//

  /// Each thread has a copy of this tile
  AFrag_NVFP4_16x64 a_frag;
  BFrag_NVFP4_64x8 b_frag;
  DFrag_F32_16x8 c_frag[TILE_WARP_M / TILE_ATOMIC_M]
                       [TILE_WARP_N / TILE_ATOMIC_N];

#pragma unroll
  for (int row = 0; row < TILE_WARP_M / TILE_ATOMIC_M; ++row) {
#pragma unroll
    for (int col = 0; col < TILE_WARP_N / TILE_ATOMIC_N; ++col) {
#pragma unroll
      for (int i = 0; i < c_frag[row][col].REGISTERS_PER_THREAD; ++i) {
        /// Init c fragment with 0.0
        c_frag[row][col].data[i] = 0.0f;
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Define shared memory for a and b operand
  //===--------------------------------------------------------------------===//

  extern __shared__ __align__(1024) NVFP4x2 smem[];
  NVFP4x2 *a_shared = smem;
  NVFP4x2 *b_shared = a_shared + STAGE * TILE_BLOCK_A_ELES / ELES_PER_NVFP4x2;
  uint32_t *a_sf_shared = reinterpret_cast<uint32_t *>(
      b_shared + STAGE * TILE_BLOCK_B_ELES / ELES_PER_NVFP4x2);
  uint32_t *b_sf_shared = a_sf_shared + STAGE * TILE_BLOCK_A_SF_ELES;

  //===--------------------------------------------------------------------===//
  // Define brrier for a and b tma
  //===--------------------------------------------------------------------===//

  __shared__ __align__(8) uint64_t empty_bar[STAGE];
  __shared__ __align__(8) uint64_t full_bar[STAGE];
  if (is_producer) {
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

  if (is_producer && tid == CONSUMER_THREAD_NUM) {
    for (int block_k = 0, stage = 0, phase = 0; block_k < K;
         block_k += TILE_BLOCK_K, ++stage) {
      int block_k_pack = block_k / ELES_PER_NVFP4x2;

      if (stage == STAGE) {
        stage = 0;
        phase ^= 1;
      }

      wait(&empty_bar[stage], phase);

      /// TMA A
      load_4d_async(a_shared + stage * TILE_BLOCK_A_ELES / ELES_PER_NVFP4x2,
                    &a_tensor_map, &full_bar[stage], 0, 0, block_m,
                    block_k_pack);
      /// TMA B
      load_4d_async(b_shared + stage * TILE_BLOCK_B_ELES / ELES_PER_NVFP4x2,
                    &b_tensor_map, &full_bar[stage], 0, 0, block_n,
                    block_k_pack);
      /// TMA A_SF
      load_4d_async(a_sf_shared + stage * TILE_BLOCK_A_SF_ELES,
                    &a_sf_tensor_map, &full_bar[stage], 0, 0, block_k / 64,
                    block_m);
      /// TMA B_SF
      load_4d_async(b_sf_shared + stage * TILE_BLOCK_B_SF_ELES,
                    &b_sf_tensor_map, &full_bar[stage], 0, 0, block_k / 64,
                    block_n);

      expect_bytes(&full_bar[stage],
                   TILE_IN_ELES / ELES_PER_NVFP4x2 * sizeof(NVFP4x2) +
                       (TILE_BLOCK_A_SF_ELES + TILE_BLOCK_B_SF_ELES) *
                           sizeof(uint32_t));
    } // end loop block_k
  }

  //===--------------------------------------------------------------------===//
  // Consumer
  //===--------------------------------------------------------------------===//

  if (is_consumer) {
    if (lane_id == 0) {
#pragma unroll
      for (int i = 0; i < STAGE; ++i) {
        arrive(&empty_bar[i], 1);
      }
    }

    //===------------------------------------------------------------------===//
    // Loop
    //===------------------------------------------------------------------===//

    for (int block_k = 0, stage = 0, phase = 0; block_k < K;
         block_k += TILE_BLOCK_K, ++stage) {
      if (stage == STAGE) {
        stage = 0;
        phase ^= 1;
      }

      wait(&full_bar[stage], phase);

#pragma unroll
      for (int warp_k = 0; warp_k < TILE_BLOCK_K; warp_k += TILE_WARP_K) {
#pragma unroll
        for (int atomic_k = 0; atomic_k < TILE_WARP_K;
             atomic_k += TILE_ATOMIC_K) {
#pragma unroll
          for (int atomic_m_cnt = 0; atomic_m_cnt < TILE_WARP_M / TILE_ATOMIC_M;
               ++atomic_m_cnt) {
            int atomic_m = atomic_m_cnt * TILE_ATOMIC_M;

            /// SF A
            uint32_t sfa0 = 0;
            {
              int row =
                  warp_m + atomic_m + ((lane_id % 4) % 2) * 8 + (lane_id / 4);
              int col = warp_k + atomic_k;
              sfa0 = a_sf_shared[stage * TILE_BLOCK_A_SF_ELES +
                                 col / 64 * TILE_BLOCK_M + row];
            }

            /// Load shared to fragment for A operand
            int *a_regs = (int *)a_frag.data;
            int a_shared_row = warp_m + atomic_m + (lane_id % 16);
            int a_shared_col =
                (warp_k + atomic_k + (lane_id / 16) * 32) / ELES_PER_NVFP4x2;
            int a_shared_row_swiz = SwizzleIndexMap<Swizzle>::get_row(
                a_shared_row, a_shared_col, sizeof(NVFP4x2));
            int a_shared_col_swiz = SwizzleIndexMap<Swizzle>::get_col(
                a_shared_row, a_shared_col, sizeof(NVFP4x2));
            uint32_t a_addr = __cvta_generic_to_shared(
                a_shared + stage * TILE_BLOCK_A_ELES / ELES_PER_NVFP4x2 +
                a_shared_row_swiz * TILE_BLOCK_K / ELES_PER_NVFP4x2 +
                a_shared_col_swiz);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                         "{%0, %1, %2, %3}, [%4];"
                         : "=r"(a_regs[0]), "=r"(a_regs[1]), "=r"(a_regs[2]),
                           "=r"(a_regs[3])
                         : "r"(a_addr));

#pragma unroll
            for (int atomic_n_cnt = 0;
                 atomic_n_cnt < TILE_WARP_N / TILE_ATOMIC_N; ++atomic_n_cnt) {
              int atomic_n = atomic_n_cnt * TILE_ATOMIC_N;

              /// SF B
              uint32_t sfb0 = 0;
              {
                int row = warp_n + atomic_n + (lane_id / 4);
                int col = warp_k + atomic_k;
                sfb0 = b_sf_shared[stage * TILE_BLOCK_B_SF_ELES +
                                   col / 64 * TILE_BLOCK_N + row];
              }

              /// Load shared to fragment for B operand
              int *b_regs = (int *)b_frag.data;
              int b_shared_row = warp_n + atomic_n + (lane_id % 8);
              int b_shared_col =
                  (warp_k + atomic_k + ((lane_id % 16) / 8) * 32) /
                  ELES_PER_NVFP4x2;
              int b_shared_row_swiz = SwizzleIndexMap<Swizzle>::get_row(
                  b_shared_row, b_shared_col, sizeof(NVFP4x2));
              int b_shared_col_swiz = SwizzleIndexMap<Swizzle>::get_col(
                  b_shared_row, b_shared_col, sizeof(NVFP4x2));
              uint32_t b_addr = __cvta_generic_to_shared(
                  b_shared + stage * TILE_BLOCK_B_ELES / ELES_PER_NVFP4x2 +
                  b_shared_row_swiz * TILE_BLOCK_K / ELES_PER_NVFP4x2 +
                  b_shared_col_swiz);
              asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                           "{%0, %1}, [%2];"
                           : "=r"(b_regs[0]), "=r"(b_regs[1])
                           : "r"(b_addr));

              /// Apply mma
              mma_sync_m16n8k64_row_col_nvfp4nvfp4f32(
                  c_frag[atomic_m_cnt][atomic_n_cnt].data, a_frag.data,
                  b_frag.data, sfa0, sfb0);
            } // end loop atomic_n
          } // end loop atomic_m
        } // end loop atomic_k
      } // end loop warp_k

      if (lane_id == 0) {
        arrive(&empty_bar[stage], 1);
      }
    } // end loop block_k

    //===------------------------------------------------------------------===//
    // Epilogue
    //===------------------------------------------------------------------===//

    //===------------------------------------------------------------------===//
    // Apply alpha
    //===------------------------------------------------------------------===//

    float scale = (*A_global_scale) * (*B_global_scale);

#pragma unroll
    for (int atomic_m_cnt = 0; atomic_m_cnt < TILE_WARP_M / TILE_ATOMIC_M;
         ++atomic_m_cnt) {
#pragma unroll
      for (int atomic_n_cnt = 0; atomic_n_cnt < TILE_WARP_N / TILE_ATOMIC_N;
           ++atomic_n_cnt) {
#pragma unroll
        for (int idx = 0;
             idx < c_frag[atomic_m_cnt][atomic_n_cnt].REGISTERS_PER_THREAD;
             ++idx) {
          c_frag[atomic_m_cnt][atomic_n_cnt].data[idx] *= scale;
        } // end loop idx
      } // end loop atomic_n_cnt
    } // end loop atomic_m_cnt

    //===------------------------------------------------------------------===//
    // Apply bias
    //===------------------------------------------------------------------===//

    if (bias) {
#pragma unroll
      for (int atomic_m_cnt = 0; atomic_m_cnt < TILE_WARP_M / TILE_ATOMIC_M;
           ++atomic_m_cnt) {
#pragma unroll
        for (int atomic_n_cnt = 0; atomic_n_cnt < TILE_WARP_N / TILE_ATOMIC_N;
             ++atomic_n_cnt) {
#pragma unroll
          for (int idx = 0;
               idx < c_frag[atomic_m_cnt][atomic_n_cnt].REGISTERS_PER_THREAD;
               ++idx) {
            int thread_n = atomic_n_cnt * TILE_ATOMIC_N +
                           c_frag[atomic_m_cnt][atomic_n_cnt].get_col_with_reg(
                               lane_id, idx);
            int n = block_n + warp_n + thread_n;
            if (n < N) {
              c_frag[atomic_m_cnt][atomic_n_cnt].data[idx] +=
                  fpoint_to_f32(bias[n]);
            }
          } // end loop idx
        } // end loop atomic_n_cnt
      } // end loop atomic_m_cnt
    }

    //===------------------------------------------------------------------===//
    // Write back to D
    //===------------------------------------------------------------------===//

    if (block_m + TILE_BLOCK_M <= M && block_n + TILE_BLOCK_N <= N) {
      // Not out-of-range

#pragma unroll
      for (int atomic_m_cnt = 0; atomic_m_cnt < TILE_WARP_M / TILE_ATOMIC_M;
           ++atomic_m_cnt) {
#pragma unroll
        for (int atomic_n_cnt = 0; atomic_n_cnt < TILE_WARP_N / TILE_ATOMIC_N;
             ++atomic_n_cnt) {
          D_TYPEx2 out;
          auto *out_bf16 = reinterpret_cast<D_TYPE *>(&out);
          {
            out_bf16[0] = f32_to_fpoint<D_TYPE>(
                c_frag[atomic_m_cnt][atomic_n_cnt].data[0]);
            out_bf16[1] = f32_to_fpoint<D_TYPE>(
                c_frag[atomic_m_cnt][atomic_n_cnt].data[1]);

            int thread_m =
                atomic_m_cnt * TILE_ATOMIC_M +
                c_frag[atomic_m_cnt][atomic_n_cnt].get_row_with_reg(lane_id, 0);
            int m = block_m + warp_m + thread_m;
            int thread_n =
                atomic_n_cnt * TILE_ATOMIC_N +
                c_frag[atomic_m_cnt][atomic_n_cnt].get_col_with_reg(lane_id, 0);
            int n = block_n + warp_n + thread_n;
            *reinterpret_cast<D_TYPEx2 *>(D + m * N + n) = out;
          }

          {
            out_bf16[0] = f32_to_fpoint<D_TYPE>(
                c_frag[atomic_m_cnt][atomic_n_cnt].data[2]);
            out_bf16[1] = f32_to_fpoint<D_TYPE>(
                c_frag[atomic_m_cnt][atomic_n_cnt].data[3]);

            int thread_m =
                atomic_m_cnt * TILE_ATOMIC_M +
                c_frag[atomic_m_cnt][atomic_n_cnt].get_row_with_reg(lane_id, 2);
            int m = block_m + warp_m + thread_m;
            int thread_n =
                atomic_n_cnt * TILE_ATOMIC_N +
                c_frag[atomic_m_cnt][atomic_n_cnt].get_col_with_reg(lane_id, 2);
            int n = block_n + warp_n + thread_n;
            *reinterpret_cast<D_TYPEx2 *>(D + m * N + n) = out;
          }
        } // end loop atomic_n_cnt
      } // end loop atomic_m_cnt
    } else {
      // Meet out-of-range

      // Check warp
      if (block_m + warp_m >= M || block_n + warp_n >= N) {
        return;
      }

#pragma unroll
      for (int atomic_m_cnt = 0; atomic_m_cnt < TILE_WARP_M / TILE_ATOMIC_M;
           ++atomic_m_cnt) {
        int atomic_m = atomic_m_cnt * TILE_ATOMIC_M;
        if (block_m + warp_m + atomic_m >= M) {
          break;
        }

#pragma unroll
        for (int atomic_n_cnt = 0; atomic_n_cnt < TILE_WARP_N / TILE_ATOMIC_N;
             ++atomic_n_cnt) {
          int atomic_n = atomic_n_cnt * TILE_ATOMIC_N;
          if (block_n + warp_n + atomic_n >= N) {
            break;
          }

#pragma unroll
          for (int idx = 0;
               idx < c_frag[atomic_m_cnt][atomic_n_cnt].REGISTERS_PER_THREAD;
               ++idx) {
            int thread_m =
                atomic_m + c_frag[atomic_m_cnt][atomic_n_cnt].get_row_with_reg(
                               lane_id, idx);
            int m = block_m + warp_m + thread_m;
            int thread_n =
                atomic_n + c_frag[atomic_m_cnt][atomic_n_cnt].get_col_with_reg(
                               lane_id, idx);
            int n = block_n + warp_n + thread_n;
            if (m >= M || n >= N) {
              continue;
            }

            D[m * N + n] = f32_to_fpoint<D_TYPE>(
                c_frag[atomic_m_cnt][atomic_n_cnt].data[idx]);
          } // end loop idx
        } // end loop atomic_n_cnt
      } // end loop atomic_m_cnt
    }
  } // end if consumer
}

void nvfp4_dot_scale_sm120(torch::Tensor &D, torch::Tensor const &A,
                           torch::Tensor const &A_sf,
                           torch::Tensor const &A_global_scale,
                           torch::Tensor const &B, torch::Tensor const &B_sf,
                           torch::Tensor const &B_global_scale,
                           c10::optional<torch::Tensor> const &bias) {
  /// Check D
  TORCH_CHECK(D.dim() == 2, "d must be a 2d matrix");
  auto const M = D.sizes()[0];
  auto const N = D.sizes()[1];
  TORCH_CHECK(D.scalar_type() == at::ScalarType::BFloat16 ||
                  D.scalar_type() == at::ScalarType::Half ||
                  D.scalar_type() == at::ScalarType::Float,
              "d must be bfloat16, half or float");
  CHECK_CONTIGUOUS(D, "d must be contiguous");

  /// Check A
  TORCH_CHECK(A.dim() == 2, "a must be a 2d matrix");
  CHECK_INPUT(A, at::ScalarType::Byte, "a");
  TORCH_CHECK(A.size(0) == M, "A size(0) must be ", M, " but see ", A.size(0));
  auto const K = A.size(1) * ELES_PER_NVFP4x2;
  TORCH_CHECK(K % 16 == 0, "A size(1)*2 must be multiple of 16 but see ", K);
  TORCH_CHECK(A_sf.dim() == 3, "scale_a must be a 3d matrix");
  CHECK_INPUT(A_sf, at::ScalarType::Float8_e4m3fn, "scale_a");
  TORCH_CHECK(A_sf.size(0) == ceil_div(K, 64));
  TORCH_CHECK(A_sf.size(1) == align_up(M, 16));
  TORCH_CHECK(A_sf.size(2) == 4);
  CHECK_INPUT(A_global_scale, at::ScalarType::Float, "global_scale_a");
  TORCH_CHECK(A_global_scale.dim() == 0, "A_global_scale must be a scalar");

  /// Check B
  TORCH_CHECK(B.dim() == 2, "b must be a 2d matrix");
  CHECK_INPUT(B, at::ScalarType::Byte, "b");
  TORCH_CHECK(B.size(1) == A.size(1), "B size(1) must equal A size(1)");
  TORCH_CHECK(B_sf.dim() == 3, "scale_b must be a 3d matrix");
  CHECK_INPUT(B_sf, at::ScalarType::Float8_e4m3fn, "scale_b");
  TORCH_CHECK(B_sf.size(0) == ceil_div(K, 64));
  TORCH_CHECK(B_sf.size(1) == align_up(N, 16));
  TORCH_CHECK(B_sf.size(2) == 4);
  CHECK_INPUT(B_global_scale, at::ScalarType::Float, "global_scale_b");
  TORCH_CHECK(B_global_scale.dim() == 0, "B_global_scale must be a scalar");

  /// Check bias
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().scalar_type() == at::ScalarType::BFloat16 ||
                    bias.value().scalar_type() == at::ScalarType::Half ||
                    bias.value().scalar_type() == at::ScalarType::Float,
                "bias must be bfloat16, half or float");
    CHECK_CONTIGUOUS(bias.value(), "bias must be contiguous");
    TORCH_CHECK(bias.value().dim() == 2, "bias must be a 2d matrix");
    TORCH_CHECK(bias.value().size(0) == 1, "bias shape[0] must be 1");
    TORCH_CHECK(bias.value().size(1) == N, "bias shape[1] must be ", N);
  }

  /// Dispatch output
  DISPATCH_AT_TENSOR_TYPES<
      at::ScalarType::BFloat16, at::ScalarType::Half,
      at::ScalarType::Float>(D.scalar_type(), [&]<at::ScalarType D_AT_TYPE>() {
    /// Dispatch bias
    auto bias_type = bias.has_value() ? bias.value().scalar_type() : D_AT_TYPE;
    DISPATCH_AT_TENSOR_TYPES<
        at::ScalarType::BFloat16, at::ScalarType::Half,
        at::ScalarType::Float>(bias_type, [&]<at::ScalarType BIAS_AT_TYPE>() {
      using D_T = typename AtDataTraits<D_AT_TYPE>::Type;
      using BIAS_T = typename AtDataTraits<BIAS_AT_TYPE>::Type;

      /// Grid & Block dim3
      dim3 grid(ceil_div(N, TILE_BLOCK_N) * ceil_div(M, TILE_BLOCK_M));
      dim3 block(CONSUMER_THREAD_NUM + PRODUCER_THREAD_NUM);

      /// Get CUDA stream
      auto stream = at::cuda::getCurrentCUDAStream().stream();

      /// Set dynamic shared memory size
      const auto SWIZZLE =
          get_swizzle<TILE_BLOCK_K / ELES_PER_NVFP4x2, sizeof(NVFP4x2)>();
      auto *kernel =
          nvfp4_dot_scale_sm120_kernel<D_AT_TYPE, BIAS_AT_TYPE, SWIZZLE>;
      size_t smem_size =
          TILE_IN_ELES / ELES_PER_NVFP4x2 * sizeof(NVFP4x2) * STAGE +
          (TILE_BLOCK_A_SF_ELES + TILE_BLOCK_B_SF_ELES) * sizeof(uint32_t) *
              STAGE;
      CHECK_CUDA_ERROR(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      /// TMA descriptor
      CUtensorMap a_tensor_map = create_4d_tensor_map<
          1, 1, TILE_BLOCK_M, TILE_BLOCK_K / ELES_PER_NVFP4x2,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8, NVFP4x2, SWIZZLE>(
          reinterpret_cast<const NVFP4x2 *>(A.data_ptr()),
          static_cast<uint64_t>(1), static_cast<uint64_t>(1),
          static_cast<uint64_t>(A.size(0)), static_cast<uint64_t>(A.size(1)));
      CUtensorMap b_tensor_map = create_4d_tensor_map<
          1, 1, TILE_BLOCK_N, TILE_BLOCK_K / ELES_PER_NVFP4x2,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8, NVFP4x2, SWIZZLE>(
          reinterpret_cast<const NVFP4x2 *>(B.data_ptr()),
          static_cast<uint64_t>(1), static_cast<uint64_t>(1),
          static_cast<uint64_t>(B.size(0)), static_cast<uint64_t>(B.size(1)));
      CUtensorMap a_sf_tensor_map = create_4d_tensor_map<
          1, 1, TILE_BLOCK_K / (BLOCK_SIZE * 4), TILE_BLOCK_M,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32, uint32_t>(
          reinterpret_cast<const uint32_t *>(A_sf.data_ptr()),
          static_cast<uint64_t>(1), static_cast<uint64_t>(1),
          static_cast<uint64_t>(A_sf.size(0)),
          static_cast<uint64_t>(A_sf.size(1)));
      CUtensorMap b_sf_tensor_map = create_4d_tensor_map<
          1, 1, TILE_BLOCK_K / (BLOCK_SIZE * 4), TILE_BLOCK_N,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32, uint32_t>(
          reinterpret_cast<const uint32_t *>(B_sf.data_ptr()),
          static_cast<uint64_t>(1), static_cast<uint64_t>(1),
          static_cast<uint64_t>(B_sf.size(0)),
          static_cast<uint64_t>(B_sf.size(1)));

      /// Launch kernel
      kernel<<<grid, block, smem_size, stream>>>(
          reinterpret_cast<D_T *>(D.data_ptr()),
          reinterpret_cast<const float *>(A_global_scale.data_ptr()),
          reinterpret_cast<const float *>(B_global_scale.data_ptr()),
          bias.has_value()
              ? reinterpret_cast<const BIAS_T *>(bias.value().data_ptr())
              : nullptr,
          M, N, K, a_tensor_map, b_tensor_map, a_sf_tensor_map,
          b_sf_tensor_map);
    });
  });
}

} // namespace ac4k
