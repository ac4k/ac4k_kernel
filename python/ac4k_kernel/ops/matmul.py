import torch
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_cuda_nvfp4_matmul():
    try:
        from ._cuda_ops import nvfp4_matmul_sm120
        return nvfp4_matmul_sm120
    except ImportError as e:
        raise ImportError(
            "CUDA operator 'nvfp4_matmul_sm120' failed to load. "
            "Possible reasons: CUDA not available, or module not compiled."
        ) from e


def nvfp4_matmul(a, b, scales_a, scales_b, alpha, bias=None, out=None):
    _nvfp4_matmul_sm120 = _load_cuda_nvfp4_matmul()

    if out is None:
        m, n = a.shape[0], b.shape[0]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    _nvfp4_matmul_sm120(out, a, b, scales_a, scales_b, alpha, bias)

    return out


@lru_cache(maxsize=1)
def _load_cuda_internal_nvfp4_matmul():
    try:
        from ._cuda_ops import _internal_nvfp4_matmul_sm120
        return _internal_nvfp4_matmul_sm120
    except ImportError as e:
        raise ImportError(
            "CUDA operator 'nvfp4_matmul_sm120' failed to load. "
            "Possible reasons: CUDA not available, or module not compiled."
        ) from e


def _internal_nvfp4_matmul(a,
                           b,
                           scales_a,
                           scales_b,
                           alpha,
                           bias=None,
                           out=None):
    _nvfp4_matmul_sm120 = _load_cuda_internal_nvfp4_matmul()

    if out is None:
        m, n = a.shape[0], b.shape[0]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    _nvfp4_matmul_sm120(out, a, b, scales_a, scales_b, alpha, bias)

    return out
