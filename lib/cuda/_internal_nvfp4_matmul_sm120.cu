#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
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

using NVFP4x2 = uint8_t;
using E4M3 = uint8_t;
constexpr int ELES_PER_NVFP4x2 = 2;

//===----------------------------------------------------------------------===//
// Matrix matmul kernel
// D bf16 dtype
// A/B nvfp4 dtype
// accumulator dtype:float
// A shape:MxK stride:Kx1 matrix, row major
// B shape:KxN stride:1xK matrix, col major
// C shape:MxN stride:Nx1 matrix, row major
//===----------------------------------------------------------------------===//

__global__ static void
nvfp4_matmul_sm120_kernel(__nv_bfloat16 *D, const NVFP4x2 *A, const NVFP4x2 *B,
                          const E4M3 *A_sf, const E4M3 *B_sf,
                          const float *alpha, const __nv_bfloat16 *bias, int M,
                          int N, int K) {
  // Use
  // mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3

  int tile_row = blockIdx.y;
  int tile_col = blockIdx.x;
  int tid = threadIdx.x;
  int block_row = tile_row * 16;
  int block_col = tile_col * 8;

  /// Each thread has a copy of this tile
  AFrag_NVFP4_16x64 a_tile;
  BFrag_NVFP4_64x8 b_tile;
  DFrag_F32_16x8 c_tile;

#pragma unroll
  for (int i = 0; i < c_tile.REGISTERS_PER_THREAD; ++i) {
    /// Initialize C(accumulator) tile to 0
    c_tile.data[i] = 0.0f;
  }

  for (int k = 0; k < K; k += 64) {
/// Load A fragment
#pragma unroll
    for (int idx = 0; idx < a_tile.REGISTERS_PER_THREAD; ++idx) {
      int row = a_tile.get_row_with_reg(tid, idx);
      int col = a_tile.get_col_with_reg(tid, idx);
      if (block_row + row >= M || k + col >= K) {
        a_tile.data[idx] = 0;
        continue;
      } else {
        const NVFP4x2 *A_tile =
            A + ((block_row + row) * K + col + k) / ELES_PER_NVFP4x2;
        a_tile.data[idx] =
            *reinterpret_cast<const AFrag_NVFP4_16x64::REG_TYPE *>(A_tile);
      }
    }

    /// Load B fragment
#pragma unroll
    for (int idx = 0; idx < b_tile.REGISTERS_PER_THREAD; ++idx) {
      int row = b_tile.get_row_with_reg(tid, idx);
      int col = b_tile.get_col_with_reg(tid, idx);
      if (row + k >= K || block_col + col >= N) {
        b_tile.data[idx] = 0;
        continue;
      } else {
        const NVFP4x2 *B_tile =
            B + (row + k + (block_col + col) * K) / ELES_PER_NVFP4x2;
        b_tile.data[idx] =
            *reinterpret_cast<const BFrag_NVFP4_64x8::REG_TYPE *>(B_tile);
      }
    }

    /// SF
    /// Layout
    /// [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)] x u8
    /// dim0,       dim1,      dim2,       dim3,      dim4
    /// SF A
    uint32_t sfa0 = 0;
    if (tid % 4 < 2 && k < K) {
      int row = block_row + ((tid % 4) % 2) * 8 + (tid / 4);
      int col = k;
      // sfa0 = reinterpret_cast<const uint32_t *>(
      //     A_sf)[(block_row + ((tid % 4) % 2) * 8 + (tid / 4)) * K / 64 +
      //           k / 64];

      int row_swiz = (row % 32) + 32 * (col / (16 * 4)) +
                     (K / (16 * 4)) * 32 * (row / 128);
      int col_swiz = ((row / 32) % 4) * 4;
      sfa0 =
          reinterpret_cast<const uint32_t *>(A_sf)[row_swiz * 4 + col_swiz / 4];
    }
    /// SF B
    uint32_t sfb0 = 0;
    if (tid % 4 == 0 && k < K) {
      // sfb0 = reinterpret_cast<const uint32_t *>(
      //     B_sf)[(block_col + (tid / 4)) * K / 64 + k / 64];
      int row = block_col + (tid / 4);
      int col = k;
      int row_swiz = (row % 32) + 32 * (col / (16 * 4)) +
                     (K / (16 * 4)) * 32 * (row / 128);
      int col_swiz = ((row / 32) % 4) * 4;
      sfb0 =
          reinterpret_cast<const uint32_t *>(B_sf)[row_swiz * 4 + col_swiz / 4];
    }

    /// Apply mma
    mma_sync_m16n8k64_row_col_nvfp4nvfp4f32(c_tile.data, a_tile.data,
                                            b_tile.data, sfa0, sfb0);
  } // end loop k

  //===----------------------------------------------------------------------===//
  // Apply alpha
  //===----------------------------------------------------------------------===//

#pragma unroll
  for (int idx = 0; idx < c_tile.REGISTERS_PER_THREAD; ++idx) {
    c_tile.data[idx] *= *alpha;
  }

  //===----------------------------------------------------------------------===//
  // Apply bias
  //===----------------------------------------------------------------------===//

  if (bias) {
#pragma unroll
    for (int idx = 0; idx < c_tile.REGISTERS_PER_THREAD; ++idx) {
      int col = c_tile.get_col_with_reg(tid, idx);
      if (col < N) {
        c_tile.data[idx] += __bfloat162float(bias[block_col + col]);
      }
    }
  }

  //===----------------------------------------------------------------------===//
  // Write back to D
  //===----------------------------------------------------------------------===//

#pragma unroll
  for (int idx = 0; idx < c_tile.REGISTERS_PER_THREAD; ++idx) {
    int row = c_tile.get_row_with_reg(tid, idx);
    int col = c_tile.get_col_with_reg(tid, idx);
    if (block_row + row >= M || block_col + col >= N) {
      continue;
    }
    D[(block_row + row) * N + block_col + col] =
        __float2bfloat16_rn(c_tile.data[idx]);
  }
}

void _internal_nvfp4_matmul_sm120(torch::Tensor &D, torch::Tensor const &A,
                                  torch::Tensor const &B,
                                  torch::Tensor const &A_sf,
                                  torch::Tensor const &B_sf,
                                  torch::Tensor const &alpha,
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
  int64_t K_ALIGN = 64;
  auto M_PAD = (M + M_ALIGN - 1) / M_ALIGN * M_ALIGN;
  auto N_PAD = (N + N_ALIGN - 1) / N_ALIGN * N_ALIGN;
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

  /// Get stream
  at::cuda::CUDAGuard device_guard{(char)A.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

  nvfp4_matmul_sm120_kernel<<<dim3(ceil_div(N, 8), ceil_div(M, 16)), dim3(32),
                              0, stream>>>(
      reinterpret_cast<__nv_bfloat16 *>(D.data_ptr()),
      reinterpret_cast<const NVFP4x2 *>(A.data_ptr()),
      reinterpret_cast<const NVFP4x2 *>(B.data_ptr()),
      reinterpret_cast<const E4M3 *>(A_sf.data_ptr()),
      reinterpret_cast<const E4M3 *>(B_sf.data_ptr()),
      reinterpret_cast<const float *>(alpha.data_ptr()),
      bias.has_value()
          ? reinterpret_cast<const __nv_bfloat16 *>(bias.value().data_ptr())
          : nullptr,
      M, N, K);
}

} // namespace ac4k
