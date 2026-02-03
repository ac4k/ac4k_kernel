#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "ac4k_kernel/ops.h"
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

__global__ void rope_3d_apply_kernel(
    const __nv_bfloat16* __restrict__ x,        // [B, S, N, D]
    const int32_t* __restrict__ grid_sizes,     // [B, 3]
    const cuDoubleComplex* __restrict__ freqs,  // [max_pos, C]
    __nv_bfloat16* __restrict__ output,         // [B, S, N, D]
    const int32_t B,
    const int32_t S,
    const int32_t N,
    const int32_t D,
    const int32_t C)                            // C = D/2
{
    // Calculate global indices
    const int32_t total_threads = gridDim.x * blockDim.x;
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate how many position-head pairs we need to process
    const int64_t total_position_heads = static_cast<int64_t>(B) * S * N;
    
    for (int64_t idx = tid; idx < total_position_heads; idx += total_threads) {
        // Decode idx into batch, position, head indices
        const int32_t head = idx % N;
        const int64_t position_head = idx / N;
        const int32_t pos = position_head % S;
        const int32_t batch = position_head / S;
         
        // Get grid dimensions for this batch
        const int32_t F = grid_sizes[batch * 3 + 0];  // Frames
        const int32_t H = grid_sizes[batch * 3 + 1];  // Height
        const int32_t W = grid_sizes[batch * 3 + 2];  // Width
        const int32_t seq_len = F * H * W;
        
        // Skip padding positions
        if (pos >= seq_len) continue;
        
        // Calculate 3D indices from linear position
        const int32_t f_idx = pos / (H * W);
        const int32_t h_idx = (pos % (H * W)) / W;
        const int32_t w_idx = pos % W;
        
        // Calculate frequency splits based on C
        const int32_t freq_c1 = C - 2 * (C / 3);
        const int32_t freq_c2 = C / 3;
        const int32_t freq_c3 = C / 3;
        
        // Base pointer for input and output for this position-head
        const int64_t base_offset = static_cast<int64_t>(batch) * S * N * D + 
                                    static_cast<int64_t>(pos) * N * D + 
                                    static_cast<int64_t>(head) * D;
        
        const __nv_bfloat16* input_ptr = x + base_offset;
        __nv_bfloat16* output_ptr = output + base_offset;
        
        // Process D elements (D/2 complex pairs)
        for (int32_t d = 0; d < D; d += 2) {
            // Read input as complex pair
            const __nv_bfloat16 x_real = input_ptr[d];
            const __nv_bfloat16 x_imag = input_ptr[d + 1];
            
            // Convert to double for calculations (to match reference implementation precision)
            const double x_r = static_cast<double>(__bfloat162float(x_real));
            const double x_i = static_cast<double>(__bfloat162float(x_imag));
            
            // Calculate frequency components from freqs
            const int32_t freq_dim = d / 2;  // Which complex pair this is (0 to C-1)
            
            // In the Python reference, freqs is split into 3 parts along dim=1:
            // freqs[0]: [max_pos, freq_c1] - temporal (frames)
            // freqs[1]: [max_pos, freq_c2] - height  
            // freqs[2]: [max_pos, freq_c3] - width
            // The freqs tensor passed to the function has shape [max_pos, C] but conceptually
            // represents these 3 concatenated tensors along dim=1
            
            cuDoubleComplex freq_dc;
            
            if (freq_dim < freq_c1) {
                // Temporal frequency: freqs[0][f_idx, freq_dim]
                // Linear: f_idx * C + 0 + freq_dim
                const int64_t tensor_index = static_cast<int64_t>(f_idx) * C + freq_dim;
                freq_dc = freqs[tensor_index];
            } else if (freq_dim < freq_c1 + freq_c2) {
                // Height frequency: freqs[1][h_idx, freq_dim - freq_c1]
                // Linear: h_idx * C + freq_c1 + (freq_dim - freq_c1)  
                const int32_t freq_idx_in_part = freq_dim - freq_c1;
                const int64_t tensor_index = static_cast<int64_t>(h_idx) * C + freq_c1 + freq_idx_in_part;
                freq_dc = freqs[tensor_index];
            } else {
                // Width frequency: freqs[2][w_idx, freq_dim - freq_c1 - freq_c2]
                // Linear: w_idx * C + freq_c1 + freq_c2 + (freq_dim - freq_c1 - freq_c2)
                const int32_t freq_idx_in_part = freq_dim - freq_c1 - freq_c2;
                const int64_t tensor_index = static_cast<int64_t>(w_idx) * C + freq_c1 + freq_c2 + freq_idx_in_part;
                freq_dc = freqs[tensor_index];
            }
            
            const double freq_r = freq_dc.x;
            const double freq_i = freq_dc.y;
            
            // Perform complex multiplication: (a_r + i*a_i) * (b_r + i*b_i) 
            // = (a_r*b_r - a_i*b_i) + i*(a_r*b_i + a_i*b_r)
            const double result_r = x_r * freq_r - x_i * freq_i;
            const double result_i = x_r * freq_i + x_i * freq_r;
            
            // Convert back to bfloat16
            output_ptr[d] = __float2bfloat16(static_cast<float>(result_r));
            output_ptr[d + 1] = __float2bfloat16(static_cast<float>(result_i));
        }
    }
}

namespace ac4k {

void rope3d(const torch::Tensor& x,           // [B, S, N, D], bfloat16
                   const torch::Tensor& grid_sizes,  // [B, 3], int32
                   const torch::Tensor& freqs,       // [max_pos, C], complex128
                   torch::Tensor& output)            // [B, S, N, D], bfloat16
{
    // Check inputs
    CHECK_INPUT(x, at::ScalarType::BFloat16, "x must be bfloat16 CUDA tensor");
    CHECK_INPUT(grid_sizes, at::ScalarType::Int, "grid_sizes must be int32 CUDA tensor");
    CHECK_INPUT(freqs, at::ScalarType::ComplexDouble, "freqs must be complex128 CUDA tensor");
    CHECK_OUTPUT(output, at::ScalarType::BFloat16, "output must be bfloat16 CUDA tensor");
    
    // Check dimensions
    TORCH_CHECK(x.dim() == 4, "x must be 4D tensor");
    TORCH_CHECK(grid_sizes.dim() == 2, "grid_sizes must be 2D tensor");
    TORCH_CHECK(freqs.dim() == 2, "freqs must be 2D tensor");
    TORCH_CHECK(output.dim() == 4, "output must be 4D tensor");
    
    // Check sizes
    const int64_t B = x.size(0);
    const int64_t S = x.size(1);
    const int64_t N = x.size(2);
    const int64_t D = x.size(3);
    const int64_t C = D / 2;  // Complex pairs
    
    TORCH_CHECK(grid_sizes.size(0) == B, "grid_sizes batch size must match x");
    TORCH_CHECK(grid_sizes.size(1) == 3, "grid_sizes must have 3 dimensions (F, H, W)");
    TORCH_CHECK(output.size(0) == B, "output batch size must match x");
    TORCH_CHECK(output.size(1) == S, "output sequence length must match x");
    TORCH_CHECK(output.size(2) == N, "output num heads must match x");
    TORCH_CHECK(output.size(3) == D, "output feature dimension must match x");
    
    // Determine grid configuration
    const int64_t total_position_heads = B * S * N;
    const int32_t threads_per_block = 256;
    const int32_t num_blocks = (total_position_heads + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    rope_3d_apply_kernel<<<num_blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const int32_t*>(grid_sizes.data_ptr<int32_t>()),
        reinterpret_cast<const cuDoubleComplex*>(freqs.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        static_cast<int32_t>(B),
        static_cast<int32_t>(S),
        static_cast<int32_t>(N),
        static_cast<int32_t>(D),
        static_cast<int32_t>(C)
    );
    
    // Check for kernel launch errors
    AT_CUDA_CHECK(cudaGetLastError());
}

} // namespace ac4k
