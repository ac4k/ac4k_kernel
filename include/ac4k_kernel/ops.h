#pragma once

#include <ATen/ATen.h>
#include <torch/extension.h>

namespace ac4k {

//===----------------------------------------------------------------------===//
// Attention kernel interface
//===----------------------------------------------------------------------===//

void mha_nvfp4_fwd(torch::Tensor &o, torch::Tensor &q, torch::Tensor &q_sf,
                   torch::Tensor &q_global_scale, torch::Tensor &k,
                   torch::Tensor &k_sf, torch::Tensor &k_global_scale,
                   torch::Tensor &v, torch::Tensor &v_sf,
                   torch::Tensor &v_global_scale, float sm_scale);

void mha_int8_x_fp8_fwd(torch::Tensor &o, torch::Tensor &q,
                         torch::Tensor &q_sf, torch::Tensor &k,
                         torch::Tensor &k_sf, torch::Tensor &v,
                         torch::Tensor &v_sf, float sm_scale);

//===----------------------------------------------------------------------===//
// Quantize kernel interface
//===----------------------------------------------------------------------===//

void quantize_nvfp4(torch::Tensor &out, torch::Tensor &sf,
                    torch::Tensor const &in, torch::Tensor const &global_scale,
                    uint32_t cross_dim, uint32_t reduce_dim, bool swizzle);

void quantize_fp8(torch::Tensor &out, torch::Tensor &sf,
                  torch::Tensor const &in, torch::Tensor const &scale_max,
                  uint32_t cross_dim, uint32_t reduce_dim, bool swizzle);

void quantize_int8(torch::Tensor &out, torch::Tensor &sf,
                   torch::Tensor const &in, torch::Tensor const &scale_max,
                   uint32_t cross_dim, uint32_t reduce_dim);

//===----------------------------------------------------------------------===//
// Linear kernel interface
//===----------------------------------------------------------------------===//

void linear_nvfp4(torch::Tensor &D, torch::Tensor const &A,
                  torch::Tensor const &A_sf,
                  torch::Tensor const &A_global_scale,
                  torch::Tensor const &B, torch::Tensor const &B_sf,
                  torch::Tensor const &B_global_scale,
                  c10::optional<torch::Tensor> const &bias);

//===----------------------------------------------------------------------===//
// RoPE kernel interface
//===----------------------------------------------------------------------===//

void rope3d(const torch::Tensor &x,          // [B, S, N, D], bfloat16
            const torch::Tensor &grid_sizes, // [B, 3], int32
            const torch::Tensor &freqs,      // [max_pos, C], complex128
            torch::Tensor &output);           // [B, S, N, D], bfloat16

} // namespace ac4k
