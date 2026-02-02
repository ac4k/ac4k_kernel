"""Linear (fully-connected layer) operators for NVFp4 precision.

Provides zero-overhead access to architecture-optimized kernels.
"""
import torch

# Direct imports - zero runtime dispatch overhead
from .._cuda_ops import linear_nvfp4 as _linear_nvfp4


# Direct function reference for zero-overhead access
linear_nvfp4 = _linear_nvfp4


class Ac4kLinearOp(torch.nn.Module):

    def __init__(self):
        super(Ac4kLinearOp, self).__init__()

    def forward(self,
                x,
                x_scale,
                x_global_scale,
                weight,
                weight_scale,
                weight_global_scale,
                bias,
                out=None):
        if (bias is not None) and (bias.ndim == 1):
            bias = bias.reshape(1, -1)

        if out is None:
            m, n = x.shape[0], weight.shape[0]
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _linear_nvfp4(out, x, x_scale, x_global_scale, weight, weight_scale, weight_global_scale,
               bias)

        return out


def linear(x,
           x_scale,
           x_global_scale,
           weight,
           weight_scale,
           weight_global_scale,
           bias=None,
           out=None):
    """
    NVFP4 Linear (fully-connected layer)

    y = x @ weight.T + bias

    Arguments:
        x: Input activation, must be nvfp4 dtype packed to uint8. Row major.
        x_scale: Input scale factor, must be fp8e4m3 dtype.
        x_global_scale: Input global scale factor, must be fp32 dtype.
        weight: Weight matrix, must be nvfp4 dtype packed to uint8. Col major (transposed).
        weight_scale: Weight scale factor, must be fp8e4m3 dtype.
        weight_global_scale: Weight global scale factor, must be fp32 dtype.
        bias: Optional bias, must be bf16, fp16 or fp32. Shape [1, N] or [N].
        out: Optional output tensor. If None, a new bf16 tensor is created.

    Return:
        out: Output tensor [M, N] in bf16 dtype.
    """
    op = Ac4kLinearOp()
    return op(x,
              x_scale,
              x_global_scale,
              weight,
              weight_scale,
              weight_global_scale,
              bias,
              out=out)
