#pragma once

#include <ATen/ATen.h>
#include <torch/extension.h>

namespace ac4k {

void nvfp4_mha_fwd_sm120(torch::Tensor &o, torch::Tensor &q,
                         torch::Tensor &q_sf, torch::Tensor &q_global_scale,
                         torch::Tensor &k, torch::Tensor &k_sf,
                         torch::Tensor &k_global_scale, torch::Tensor &v,
                         torch::Tensor &v_sf, torch::Tensor &v_global_scale,
                         float sm_scale);

void qk_int8_pv_fp8_mha_fwd_sm120(torch::Tensor &o, torch::Tensor &q,
                                  torch::Tensor &q_sf, torch::Tensor &k,
                                  torch::Tensor &k_sf, torch::Tensor &v,
                                  torch::Tensor &v_sf, float sm_scale);

void nvfp4_quantize_sm120(torch::Tensor &out, torch::Tensor &sf,
                          torch::Tensor const &in,
                          torch::Tensor const &rcp_global_scale,
                          uint32_t cross_dim, uint32_t reduce_dim,
                          bool swizzle);

void fp8_quantize_sm120(torch::Tensor &out, torch::Tensor &sf,
                        torch::Tensor const &in, torch::Tensor const &scale_max,
                        uint32_t cross_dim, uint32_t reduce_dim, bool swizzle);

void int8_quantize_sm120(torch::Tensor &out, torch::Tensor &sf,
                         torch::Tensor const &in,
                         torch::Tensor const &scale_max, uint32_t cross_dim,
                         uint32_t reduce_dim);

void nvfp4_quant_sm120(torch::Tensor &output, torch::Tensor &output_sf,
                       torch::Tensor const &input,
                       torch::Tensor const &input_global_scale);

void nvfp4_matmul_sm120(torch::Tensor &D, torch::Tensor const &A,
                        torch::Tensor const &B, torch::Tensor const &A_sf,
                        torch::Tensor const &B_sf, torch::Tensor const &alpha,
                        c10::optional<torch::Tensor> const &bias);

void _internal_nvfp4_matmul_sm120(torch::Tensor &D, torch::Tensor const &A,
                                  torch::Tensor const &B,
                                  torch::Tensor const &A_sf,
                                  torch::Tensor const &B_sf,
                                  torch::Tensor const &alpha,
                                  c10::optional<torch::Tensor> const &bias);

void nvfp4_dot_scale_sm120(torch::Tensor &D, torch::Tensor const &A,
                           torch::Tensor const &A_sf,
                           torch::Tensor const &A_global_scale,
                           torch::Tensor const &B, torch::Tensor const &B_sf,
                           torch::Tensor const &B_global_scale,
                           c10::optional<torch::Tensor> const &bias);

void rope_3d_apply(const torch::Tensor &x,          // [B, S, N, D], bfloat16
                   const torch::Tensor &grid_sizes, // [B, 3], int32
                   const torch::Tensor &freqs,      // [max_pos, C], complex128
                   torch::Tensor &output);          // [B, S, N, D], bfloat16

} // namespace ac4k
