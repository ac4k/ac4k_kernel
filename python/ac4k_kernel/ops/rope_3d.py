"""
3D Rotary Position Embedding (RoPE) operators.

Provides zero-overhead access to architecture-optimized kernels.
"""
from typing import Optional
import torch

# Direct imports - zero runtime dispatch overhead
from .._cuda_ops import rope3d as _rope3d


# Direct function reference for zero-overhead access
rope3d_kernel = _rope3d


class Ac4kRoPEOp(torch.nn.Module):

    def __init__(self):
        super(Ac4kRoPEOp, self).__init__()

    def forward(self, x, grid_sizes, freqs, output=None):
        if output is None:
            output = torch.empty_like(x)

        # Call the CUDA implementation
        _rope3d(x, grid_sizes, freqs, output)

        return output


def rope3d(
    x: torch.Tensor,  # [B, S, N, D], bfloat16
    grid_sizes: torch.Tensor,  # [B, 3], int32
    freqs: torch.Tensor,  # [max_pos, C], complex128
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

    op = Ac4kRoPEOp()
    return op(x, grid_sizes, freqs, output=output)
