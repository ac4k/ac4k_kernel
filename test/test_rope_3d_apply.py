import torch
import numpy as np
from ac4k_kernel.ops import rope3d

def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

def rope_apply_reference(x, grid_sizes, freqs, data_type=torch.bfloat16):
    """Reference implementation from the task description."""
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        x_i = x_i.to(data_type)
        # append to collection
        output.append(x_i)
    return torch.stack(output)


def test_rope_3d_apply():
    """Test the 3D RoPE implementation with the specified test case."""
    print("Testing 3D RoPE implementation...")
    
    # Test case parameters
    B, F, H, W = 1, 21, 45, 80
    S = F * H * W  # 75600
    N, D = 40, 128
    C = D // 2  # 64
    max_pos = 1024
    
    print(f"Test dimensions:")
    print(f"  Batch size (B): {B}")
    print(f"  Sequence length (S): {S} = {F} × {H} × {W}")
    print(f"  Heads (N): {N}")
    print(f"  Feature dimension (D): {D}")
    print(f"  Complex pairs (C): {C}")
    print(f"  Max positions: {max_pos}")
    
    # Create input tensor [B, S, N, D] in bfloat16
    print("\nCreating input tensors...")
    x = torch.randn(B, S, N, D, dtype=torch.bfloat16, device='cuda')
    print(f"x shape: {x.shape}, dtype: {x.dtype}")
    
    # Create grid sizes [B, 3] in int32
    grid_sizes = torch.tensor([[F, H, W]], dtype=torch.int32, device='cuda')
    print(f"grid_sizes shape: {grid_sizes.shape}, content: {grid_sizes}")
    
    # Create freqs [max_pos, C] using rope_params
    # Note: rope_params expects `dim` and returns complex freqs with shape [max_seq_len, dim//2]
    freqs = rope_params(max_pos, D).to(device='cuda')
    print(f"freqs shape: {freqs.shape}, dtype: {freqs.dtype}")
    
    # Create output tensor
    output = torch.empty(B, S, N, D, dtype=torch.bfloat16, device='cuda')
    print(f"output shape: {output.shape}, dtype: {output.dtype}")
    
    # Run CUDA implementation
    print("\nRunning CUDA implementation...")
    try:
        result_cuda = rope3d(x, grid_sizes, freqs, output)
        print("✓ CUDA implementation completed successfully")
    except Exception as e:
        print(f"✗ CUDA implementation failed: {e}")
        return False
    
    # Run reference implementation on CUDA (avoid transferring large tensors to CPU)
    print("\nRunning reference implementation...")
    try:
        result_ref_cuda = rope_apply_reference(x, grid_sizes, freqs, data_type=torch.bfloat16)
        print("✓ Reference implementation completed successfully")
    except Exception as e:
        print(f"✗ Reference implementation failed: {e}")
        return False
    
    # Compare results
    print("\nComparing results...")
    
    # Check shapes match
    if result_cuda.shape != result_ref_cuda.shape:
        print(f"✗ Shape mismatch: {result_cuda.shape} vs {result_ref_cuda.shape}")
        return False
    
    # Check data types match
    if result_cuda.dtype != result_ref_cuda.dtype:
        print(f"✗ Dtype mismatch: {result_cuda.dtype} vs {result_ref_cuda.dtype}")
        return False
    
    # Compare values using PyTorch built-in floating-point comparison
    # bfloat16 outputs can differ by ~1-2 ulp; compare in float32 with a small absolute tolerance.
    try:
        torch.testing.assert_close(
            result_cuda.float(),
            result_ref_cuda.float(),
            rtol=0.0,
            atol=1e-2,
        )
        print("✓ Results match within tolerance")
    except AssertionError as e:
        print("✗ Results differ too much")
        print(e)
        return False

    return True


if __name__ == "__main__":
    success = test_rope_3d_apply()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed!")
        exit(1)
