from typing import Optional
import torch
from . import _cuda_ops


def rope_3d_apply(
    x: torch.Tensor,           # [B, S, N, D], bfloat16
    grid_sizes: torch.Tensor,  # [B, 3], int32
    freqs: torch.Tensor,       # [max_pos, C], complex128
    output: Optional[torch.Tensor] = None  # [B, S, N, D], bfloat16
) -> torch.Tensor:
    """
    3D Rotary Position Embedding (RoPE) for transformer models.
    
    Args:
        x: Input tensor of shape [B, S, N, D] in bfloat16
        grid_sizes: 3D grid dimensions of shape [B, 3] (frames, height, width)
        freqs: Precomputed frequency tensor of shape [max_pos, D//2] in complex128
        output: Optional output tensor. If None, will be allocated.
    
    Returns:
        Output tensor of shape [B, S, N, D] with 3D positional encoding applied
    """
    if output is None:
        output = torch.empty_like(x)
    
    # Call the CUDA implementation
    _cuda_ops.rope_3d_apply(x, grid_sizes, freqs, output)
    
    return output