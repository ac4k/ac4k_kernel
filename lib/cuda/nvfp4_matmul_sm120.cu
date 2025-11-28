#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
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

//===----------------------------------------------------------------------===//
// Define tile size for block level, warp level and atomic level
//===----------------------------------------------------------------------===//

/// Block level tile size
const int TILE_BLOCK_M = 128;
const int TILE_BLOCK_N = 128;
const int TILE_BLOCK_K = 128;
const int TILE_BLOCK_PACK_K = TILE_BLOCK_K / ELES_PER_NVFP4x2;

/// Warp level tile size
const int TILE_WARP_M = 64;
const int TILE_WARP_N = 64;
const int TILE_WARP_K = 64;
const int TILE_WARP_PACK_K = TILE_WARP_K / ELES_PER_NVFP4x2;

/// Atomic level tile size
/// mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
const int TILE_ATOMIC_M = 16;
const int TILE_ATOMIC_N = 8;
const int TILE_ATOMIC_K = 64;
const int TILE_ATOMIC_PACK_K = TILE_ATOMIC_K / ELES_PER_NVFP4x2;

//===----------------------------------------------------------------------===//
// Create tma tensor map
//===----------------------------------------------------------------------===//

template <int BlockMajorSize, int BlockMinorSize>
__forceinline__ CUtensorMap create_tensor_map(const NVFP4x2 *gmem_ptr,
                                              int global_major_size,
                                              int global_minor_size) {
  CUtensorMap template_tensor_map{};

  void *gmem_address = (void *)gmem_ptr;
  uint64_t gmem_prob_shape[5] = {static_cast<uint64_t>(global_minor_size),
                                 static_cast<uint64_t>(global_major_size), 1, 1,
                                 1};
  uint64_t gmem_prob_stride[5] = {sizeof(NVFP4x2),
                                  sizeof(NVFP4x2) * global_minor_size, 0, 0, 0};
  uint32_t smem_box_shape[5] = {static_cast<uint64_t>(BlockMinorSize),
                                static_cast<uint64_t>(BlockMajorSize), 1, 1, 1};
  uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

  cuTensorMapEncodeTiled(
      &template_tensor_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, gmem_address,
      gmem_prob_shape, gmem_prob_stride + 1, smem_box_shape, smem_box_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  return template_tensor_map;
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

__global__ static void nvfp4_matmul_sm120_kernel(
    __nv_bfloat16 *D, const NVFP4x2 *A, const NVFP4x2 *B, const E4M3 *A_sf,
    const E4M3 *B_sf, const float *alpha, const __nv_bfloat16 *bias, int M,
    int N, int K, const __grid_constant__ CUtensorMap a_tensor_map,
    const __grid_constant__ CUtensorMap b_tensor_map) {
  //===----------------------------------------------------------------------===//
  // Define fragment for a, b and c operand
  //===----------------------------------------------------------------------===//

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

  //===----------------------------------------------------------------------===//
  // Define shared memory for a and b operand
  //===----------------------------------------------------------------------===//

  __shared__ alignas(512) NVFP4x2 a_shared[TILE_BLOCK_M][TILE_BLOCK_PACK_K];
  __shared__ alignas(512) NVFP4x2 b_shared[TILE_BLOCK_N][TILE_BLOCK_PACK_K];

  //===----------------------------------------------------------------------===//
  // Define brrier for a and b tma
  //===----------------------------------------------------------------------===//

  __shared__ barrier a_bar;
  __shared__ barrier b_bar;
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    init(&a_bar, blockDim.x * blockDim.y * blockDim.z);
    init(&b_bar, blockDim.x * blockDim.y * blockDim.z);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  //===----------------------------------------------------------------------===//
  // Define brrier token for a and b tma
  //===----------------------------------------------------------------------===//

  barrier::arrival_token a_token, b_token;

  //===----------------------------------------------------------------------===//
  // Loop
  //===----------------------------------------------------------------------===//

  int block_m = blockIdx.y * TILE_BLOCK_M;
  int block_n = blockIdx.x * TILE_BLOCK_N;
  int warp_id_m = threadIdx.z;
  int warp_id_n = threadIdx.y;
  int warp_m = warp_id_m * TILE_WARP_M;
  int warp_n = warp_id_n * TILE_WARP_N;
  int lane_id = threadIdx.x;

  for (int block_k = 0; block_k < K; block_k += TILE_BLOCK_K) {
    int block_padk_k = block_k / ELES_PER_NVFP4x2;

    /// Load a/b from global to shared memory
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      cde::cp_async_bulk_tensor_2d_global_to_shared(
          a_shared, &a_tensor_map, block_padk_k, block_m, a_bar);
      a_token = cuda::device::barrier_arrive_tx(a_bar, 1, sizeof(a_shared));
      cde::cp_async_bulk_tensor_2d_global_to_shared(
          b_shared, &b_tensor_map, block_padk_k, block_n, b_bar);
      b_token = cuda::device::barrier_arrive_tx(b_bar, 1, sizeof(b_shared));
    } else {
      a_token = a_bar.arrive();
      b_token = b_bar.arrive();
    }

    /// Wait for load to complete
    a_bar.wait(std::move(a_token));
    b_bar.wait(std::move(b_token));

#pragma unroll
    for (int warp_k = 0; warp_k < TILE_BLOCK_K; warp_k += TILE_WARP_K) {
#pragma unroll
      for (int atomic_k = 0; atomic_k < TILE_WARP_K;
           atomic_k += TILE_ATOMIC_K) {
#pragma unroll
        for (int atomic_m_cnt = 0; atomic_m_cnt < TILE_WARP_M / TILE_ATOMIC_M;
             ++atomic_m_cnt) {
          int atomic_m = atomic_m_cnt * TILE_ATOMIC_M;

#pragma unroll
          for (int atomic_n_cnt = 0; atomic_n_cnt < TILE_WARP_N / TILE_ATOMIC_N;
               ++atomic_n_cnt) {
            int atomic_n = atomic_n_cnt * TILE_ATOMIC_N;

            /// SF
            /// Layout
            /// [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)] x u8
            /// dim0,       dim1,      dim2,       dim3,      dim4
            /// TODO(jian.wu): need handle K padding
            /// SF A
            uint32_t sfa0 = 0;
            if (lane_id % 4 < 2) {
              int row = block_m + warp_m + atomic_m + ((lane_id % 4) % 2) * 8 +
                        (lane_id / 4);
              int col = block_k + warp_k + atomic_k;

              int row_swiz = (row % 32) + 32 * (col / (16 * 4)) +
                             (K / (16 * 4)) * 32 * (row / 128);
              int col_swiz = ((row / 32) % 4) * 4;
              sfa0 = reinterpret_cast<const uint32_t *>(
                  A_sf)[row_swiz * 4 + col_swiz / 4];
            }
            /// SF B
            uint32_t sfb0 = 0;
            if (lane_id % 4 == 0) {
              int row = block_n + warp_n + atomic_n + (lane_id / 4);
              int col = block_k + warp_k + atomic_k;
              int row_swiz = (row % 32) + 32 * (col / (16 * 4)) +
                             (K / (16 * 4)) * 32 * (row / 128);
              int col_swiz = ((row / 32) % 4) * 4;
              sfb0 = reinterpret_cast<const uint32_t *>(
                  B_sf)[row_swiz * 4 + col_swiz / 4];
            }

            /// Load shared to fragment
            int *a_regs = (int *)a_frag.data;
            int *b_regs = (int *)b_frag.data;
            uint32_t a_addr = __cvta_generic_to_shared(
                &a_shared[warp_m + atomic_m + (lane_id % 16)]
                         [(warp_k + atomic_k + (lane_id / 16) * 32) /
                          ELES_PER_NVFP4x2]);
            uint32_t b_addr = __cvta_generic_to_shared(
                &b_shared[warp_n + atomic_n + (lane_id % 8)]
                         [(warp_k + atomic_k + ((lane_id % 16) / 8) * 32) /
                          ELES_PER_NVFP4x2]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                         "{%0, %1, %2, %3}, [%4];"
                         : "=r"(a_regs[0]), "=r"(a_regs[1]), "=r"(a_regs[2]),
                           "=r"(a_regs[3])
                         : "r"(a_addr));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                         "{%0, %1}, [%2];"
                         : "=r"(b_regs[0]), "=r"(b_regs[1])
                         : "r"(b_addr));

            /// Apply mma
            fma(c_frag[atomic_m_cnt][atomic_n_cnt].data[0],
                c_frag[atomic_m_cnt][atomic_n_cnt].data[1],
                c_frag[atomic_m_cnt][atomic_n_cnt].data[2],
                c_frag[atomic_m_cnt][atomic_n_cnt].data[3], a_frag.data[0],
                a_frag.data[1], a_frag.data[2], a_frag.data[3], b_frag.data[0],
                b_frag.data[1], c_frag[atomic_m_cnt][atomic_n_cnt].data[0],
                c_frag[atomic_m_cnt][atomic_n_cnt].data[1],
                c_frag[atomic_m_cnt][atomic_n_cnt].data[2],
                c_frag[atomic_m_cnt][atomic_n_cnt].data[3], sfa0, sfb0);
          } // end loop atomic_n_cnt
        } // end loop atomic_m_cnt
      } // end loop atomic_k
    } // end loop warp_k

    __syncthreads();
  } // end loop block_k

  //===----------------------------------------------------------------------===//
  // Apply alpha
  //===----------------------------------------------------------------------===//

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
        c_frag[atomic_m_cnt][atomic_n_cnt].data[idx] *= *alpha;
      } // end loop idx
    } // end loop atomic_n_cnt
  } // end loop atomic_m_cnt

  //===----------------------------------------------------------------------===//
  // Apply bias
  //===----------------------------------------------------------------------===//

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
          int thread_n =
              atomic_n_cnt * TILE_ATOMIC_N +
              c_frag[atomic_m_cnt][atomic_n_cnt].get_col_with_reg(lane_id, idx);
          int n = block_n + warp_n + thread_n;
          if (n < N) {
            c_frag[atomic_m_cnt][atomic_n_cnt].data[idx] +=
                __bfloat162float(bias[n]);
          }
        } // end loop idx
      } // end loop atomic_n_cnt
    } // end loop atomic_m_cnt
  }

  //===----------------------------------------------------------------------===//
  // Write back to D
  //===----------------------------------------------------------------------===//

  if (block_m + TILE_BLOCK_M <= M && block_n + TILE_BLOCK_N <= N) {
    // Not out-of-range

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
          int thread_m =
              atomic_m_cnt * TILE_ATOMIC_M +
              c_frag[atomic_m_cnt][atomic_n_cnt].get_row_with_reg(lane_id, idx);
          int m = block_m + warp_m + thread_m;
          if (m >= M) {
            continue;
          }
          int thread_n =
              atomic_n_cnt * TILE_ATOMIC_N +
              c_frag[atomic_m_cnt][atomic_n_cnt].get_col_with_reg(lane_id, idx);
          int n = block_n + warp_n + thread_n;
          D[m * N + n] =
              __float2bfloat16_rn(c_frag[atomic_m_cnt][atomic_n_cnt].data[idx]);
        } // end loop idx
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
              atomic_m +
              c_frag[atomic_m_cnt][atomic_n_cnt].get_row_with_reg(lane_id, idx);
          int m = block_m + warp_m + thread_m;
          int thread_n =
              atomic_n +
              c_frag[atomic_m_cnt][atomic_n_cnt].get_col_with_reg(lane_id, idx);
          int n = block_n + warp_n + thread_n;
          if (m >= M || n >= N) {
            continue;
          }

          D[m * N + n] =
              __float2bfloat16_rn(c_frag[atomic_m_cnt][atomic_n_cnt].data[idx]);
        } // end loop idx
      } // end loop atomic_n_cnt
    } // end loop atomic_m_cnt
  }
}

void nvfp4_matmul_sm120(torch::Tensor &D, torch::Tensor const &A,
                        torch::Tensor const &B, torch::Tensor const &A_sf,
                        torch::Tensor const &B_sf, torch::Tensor const &alpha,
                        c10::optional<torch::Tensor> const &bias) {
  /// Check type
  CHECK_INPUT(A_sf, at::ScalarType::Float8_e4m3fn, "scale_a");
  CHECK_INPUT(B_sf, at::ScalarType::Float8_e4m3fn, "scale_b");
  CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");

  CHECK_INPUT(A, at::ScalarType::Byte, "a");
  CHECK_INPUT(B, at::ScalarType::Byte, "b");
  CHECK_INPUT(D, at::ScalarType::BFloat16, "d");

  /// Check rank
  TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  TORCH_CHECK(B.dim() == 2, "b must be a matrix");
  TORCH_CHECK(D.dim() == 2, "d must be a matrix");

  /// Get shape
  auto const M = D.sizes()[0];
  auto const N = D.sizes()[1];
  auto const K = A.sizes()[1] * 2;

  /// Check shape, stride
  /// Check A operand: shape:MxK stride:Kx1
  TORCH_CHECK(A.sizes()[0] == M, "A shape[0] must be ", M);
  TORCH_CHECK(A.sizes()[1] == K / 2, "A shape[1] must be ", K / 2);
  TORCH_CHECK(A.stride(0) == K / 2, "A stride[0] must be ", K / 2);
  TORCH_CHECK(A.stride(1) == 1, "A stride[1] must be 1");

  /// Check B operand: shape:NxK stride:Kx1
  TORCH_CHECK(B.sizes()[0] == N, "B shape[0] must be ", N);
  TORCH_CHECK(B.sizes()[1] == K / 2, "B shape[1] must be ", K / 2);
  TORCH_CHECK(B.stride(0) == K / 2, "B stride[0] must be ", K / 2);
  TORCH_CHECK(B.stride(1) == 1, "B stride[1] must be ", 1);

  /// Check scale_a operand
  TORCH_CHECK(A_sf.dim() == 2, "scale_a must be a matrix");
  TORCH_CHECK(A_sf.sizes()[1] == K / 16, "scale_a shape[1] must be ", K / 16);
  TORCH_CHECK(A_sf.stride(0) == K / 16, "scale_a stride[0] must be ", K / 16);
  TORCH_CHECK(A_sf.stride(1) == 1, "scale_a stride[1] must be 1");

  /// Check scale_b operand
  TORCH_CHECK(B_sf.dim() == 2, "scale_b must be a matrix");
  TORCH_CHECK(B_sf.sizes()[1] == K / 16, "scale_b shape[1] must be ", K / 16);
  TORCH_CHECK(B_sf.stride(0) == K / 16, "scale_b stride[0] must be ", K / 16);
  TORCH_CHECK(B_sf.stride(1) == 1, "scale_b stride[1] must be 1");

  /// Check alighment
  int64_t M_ALIGN = 128;
  int64_t N_ALIGN = 128;
  int64_t K_ALIGN = TILE_BLOCK_K;
  auto M_PAD = align_up(M, M_ALIGN);
  auto N_PAD = align_up(N, N_ALIGN);
  TORCH_CHECK(A_sf.sizes()[0] == M_PAD, "scale_a shape[0] must be ", M_PAD);
  TORCH_CHECK(B_sf.sizes()[0] == N_PAD, "scale_b shape[0] must be ", N_PAD);
  TORCH_CHECK(K % K_ALIGN == 0, "K must be aligned with ", K_ALIGN);

  /// Check bias
  if (bias.has_value()) {
    CHECK_INPUT(bias.value(), at::ScalarType::BFloat16, "bias");
    TORCH_CHECK(bias.value().dim() == 2, "bias must be a matrix");
    TORCH_CHECK(bias.value().sizes()[0] == 1, "bias shape[0] must be 1");
    TORCH_CHECK(bias.value().sizes()[1] == N, "bias shape[1] must be ", N);
    TORCH_CHECK(bias.value().stride(0) == N, "bias stride[0] must be ", N);
    TORCH_CHECK(bias.value().stride(1) == 1, "bias stride[1] must be 1");
  }

  /// Check alpha
  CHECK_INPUT(alpha, at::ScalarType::Float, "a");
  TORCH_CHECK(alpha.dim() == 0, "alpha must be a scalar");

  /// Grid & Block dim3
  dim3 grid(ceil_div(N, TILE_BLOCK_N), ceil_div(M, TILE_BLOCK_M));
  dim3 block(32, ceil_div(TILE_BLOCK_N, TILE_WARP_N),
             ceil_div(TILE_BLOCK_M, TILE_WARP_M));

  /// Get stream
  at::cuda::CUDAGuard device_guard{(char)A.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

  /// TMA descriptor
  CUtensorMap a_tensor_map =
      create_tensor_map<TILE_BLOCK_M, TILE_BLOCK_K / ELES_PER_NVFP4x2>(
          reinterpret_cast<const NVFP4x2 *>(A.data_ptr()), M,
          K / ELES_PER_NVFP4x2);
  CUtensorMap b_tensor_map =
      create_tensor_map<TILE_BLOCK_N, TILE_BLOCK_K / ELES_PER_NVFP4x2>(
          reinterpret_cast<const NVFP4x2 *>(B.data_ptr()), N,
          K / ELES_PER_NVFP4x2);

  /// Launch kernel
  nvfp4_matmul_sm120_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16 *>(D.data_ptr()),
      reinterpret_cast<const NVFP4x2 *>(A.data_ptr()),
      reinterpret_cast<const NVFP4x2 *>(B.data_ptr()),
      reinterpret_cast<const E4M3 *>(A_sf.data_ptr()),
      reinterpret_cast<const E4M3 *>(B_sf.data_ptr()),
      reinterpret_cast<const float *>(alpha.data_ptr()),
      bias.has_value()
          ? reinterpret_cast<const __nv_bfloat16 *>(bias.value().data_ptr())
          : nullptr,
      M, N, K, a_tensor_map, b_tensor_map);
}

} // namespace ac4k
