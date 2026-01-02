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
              a_scale,
              a_global_scale,
              b,
              b_scale,
              b_global_scale,
              bias=None,
              out=None):
    kernel = _load_cuda_nvfp4_dot_scale()

    if out is None:
        m, n = a.shape[0], b.shape[0]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    kernel(out, a, a_scale, a_global_scale, b, b_scale, b_global_scale, bias)

    return out
