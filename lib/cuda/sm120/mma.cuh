#pragma once

#include <cuda.h>
#include <mma.h>

namespace ac4k::sm120 {

enum class MMAAccumulateMode {
  kInit = 0U,
  kInplace = 1U,
};

template <MMAAccumulateMode mode = MMAAccumulateMode::kInplace>
__forceinline__ __device__ void
mma_sync_m16n8k64_row_col_nvfp4nvfp4f32(float *C, uint32_t *A, uint32_t *B,
                                        uint32_t const &sfa,
                                        uint32_t const &sfb) {
  static constexpr uint16_t tidA = 0;
  static constexpr uint16_t bidA = 0;
  static constexpr uint16_t tidB = 0;
  static constexpr uint16_t bidB = 0;

  if constexpr (mode == MMAAccumulateMode::kInplace) {
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11, %12, %13},"
                 "{%14},"
                 "{%15, %16},"
                 "{%17},"
                 "{%18, %19};\n"
                 : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]),
                   "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]),
                   "r"(sfa), "h"(bidA), "h"(tidA), "r"(sfb), "h"(bidB),
                   "h"(tidB));
  } else {
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11, %12, %13},"
                 "{%14},"
                 "{%15, %16},"
                 "{%17},"
                 "{%18, %19};\n"
                 : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]),
                   "r"(B[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f),
                   "r"(sfa), "h"(bidA), "h"(tidA), "r"(sfb), "h"(bidB),
                   "h"(tidB));
  }
}

template <MMAAccumulateMode mode = MMAAccumulateMode::kInplace>
__forceinline__ __device__ void
mma_sync_m16n8k32_row_col_fp8fp8f16(uint32_t *C_U32, uint32_t *A, uint32_t *B) {
  if constexpr (mode == MMAAccumulateMode::kInplace) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
                 "{%0,  %1},"
                 "{%2,  %3,  %4,  %5},"
                 "{%6,  %7},"
                 "{%8,  %9};\n"
                 : "=r"(C_U32[0]), "=r"(C_U32[1])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]),
                   "r"(B[1]), "r"(C_U32[0]), "r"(C_U32[1]));
  } else {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
                 "{%0,  %1},"
                 "{%2,  %3,  %4,  %5},"
                 "{%6,  %7},"
                 "{%8,  %9};\n"
                 : "=r"(C_U32[0]), "=r"(C_U32[1])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]),
                   "r"(B[1]), "r"(0), "r"(0));
  }
}

template <MMAAccumulateMode mode = MMAAccumulateMode::kInplace>
__forceinline__ __device__ void
mma_sync_m16n8k32_row_col_i8i8i32(int32_t *C_I32, uint32_t *A, uint32_t *B) {
  if constexpr (mode == MMAAccumulateMode::kInplace) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C_I32[0]), "=r"(C_I32[1]), "=r"(C_I32[2]), "=r"(C_I32[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C_I32[0]), "r"(C_I32[1]), "r"(C_I32[2]), "r"(C_I32[3]));
  } else {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11, %12, %13};\n"
                 : "=r"(C_I32[0]), "=r"(C_I32[1]), "=r"(C_I32[2]),
                   "=r"(C_I32[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]),
                   "r"(B[1]), "r"(0), "r"(0), "r"(0), "r"(0));
  }
}

} // namespace ac4k::sm120
