import torch
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_cuda_nvfp4_dot_scale():
    try:
        from ._cuda_ops import nvfp4_dot_scale_sm120
        return nvfp4_dot_scale_sm120
    except ImportError as e:
        raise ImportError(
            "CUDA operator 'nvfp4_dot_scale_sm120' failed to load. "
            "Possible reasons: CUDA not available, or module not compiled."
        ) from e


class Ac4kDotScaleOp(torch.nn.Module):

    def __init__(self):
        super(Ac4kDotScaleOp, self).__init__()

    def forward(self,
                a,
                a_scale,
                a_global_scale,
                b,
                b_scale,
                b_global_scale,
                bias,
                out=None):
        kernel = _load_cuda_nvfp4_dot_scale()

        if out is None:
            m, n = a.shape[0], b.shape[0]
            out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
        kernel(out, a, a_scale, a_global_scale, b, b_scale, b_global_scale,
               bias)

        return out


def dot_scale(a,
              a_scale,
              a_global_scale,
              b,
              b_scale,
              b_global_scale,
              bias=None,
              out=None):
    """
    nvfp4 dot scale

    Arguments:
        a: dot lhs operand, must be nvfp4 dtype and pack to uint8 dtype. Must be row major.
        a_scale: dot lhs scale factor, must be fp8e4m3 dtype.
        a_global_scale: dot lhs global scale factor, must be fp32 dtype.
        b: dot rhs operand, must be nvfp4 dtype and pack to uint8 dtype. Must be col major.
        b_scale: dot rhs scale factor, must be fp8e4m3 dtype.
        b_global_scale: dot rhs global scale factor, must be fp32 dtype.
        bias: dot bias, must be same type with output.
        out: dot output. If None, a new tensor is created, which is bf16 dtype.

    Return:
        out: dot output. If out is None, a new tensor is created, which is bf16 dtype.
    """
    op = Ac4kDotScaleOp()
    return op(a,
              a_scale,
              a_global_scale,
              b,
              b_scale,
              b_global_scale,
              bias,
              out=out)
