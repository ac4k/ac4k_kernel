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


def dot_scale(a,
              scale_a,
              global_scale_a,
              b,
              scale_b,
              global_scale_b,
              bias=None,
              out=None):
    kernel = _load_cuda_nvfp4_dot_scale()

    if out is None:
        m, n = a.shape[0], b.shape[0]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    kernel(out, a, b, scale_a, scale_b, global_scale_a * global_scale_b, bias)

    return out
