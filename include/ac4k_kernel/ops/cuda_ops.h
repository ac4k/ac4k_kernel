#pragma once

#include <ATen/ATen.h>
#include <torch/extension.h>

namespace ac4k {

void nvfp4_quant_sm120(torch::Tensor &output, torch::Tensor &output_sf,
                       torch::Tensor const &input,
                       torch::Tensor const &input_global_scale);

void nvfp4_matmul_sm120(torch::Tensor &D, torch::Tensor const &A,
                        torch::Tensor const &B, torch::Tensor const &A_sf,
                        torch::Tensor const &B_sf, torch::Tensor const &alpha,
                        c10::optional<torch::Tensor> const &bias);

} // namespace ac4k
