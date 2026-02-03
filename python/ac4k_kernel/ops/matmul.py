import torch
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_cuda_nvfp4_matmul():
    try:
        from .._cuda_ops import nvfp4_matmul
        return nvfp4_matmul
    except ImportError as e:
        raise ImportError(
            "CUDA operator 'nvfp4_matmul' failed to load. "
            "Possible reasons: CUDA not available, or module not compiled."
        ) from e


def nvfp4_matmul(a, b, scales_a, scales_b, alpha, bias=None, out=None):
    _nvfp4_matmul = _load_cuda_nvfp4_matmul()

    if out is None:
        m, n = a.shape[0], b.shape[0]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    _nvfp4_matmul(out, a, b, scales_a, scales_b, alpha, bias)

    return out


@lru_cache(maxsize=1)
def _load_cuda_internal_nvfp4_matmul():
    try:
        from .._cuda_ops import _internal_nvfp4_matmul
        return _internal_nvfp4_matmul
    except ImportError as e:
        raise ImportError(
            "CUDA operator '_internal_nvfp4_matmul' failed to load. "
            "Possible reasons: CUDA not available, or module not compiled."
        ) from e


def _internal_nvfp4_matmul(a,
                           b,
                           scales_a,
                           scales_b,
                           alpha,
                           bias=None,
                           out=None):
    _fn = _load_cuda_internal_nvfp4_matmul()

    if out is None:
        m, n = a.shape[0], b.shape[0]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    _fn(out, a, b, scales_a, scales_b, alpha, bias)

    return out
