from ac4k_kernel.ops import quantize

from functools import lru_cache
import torch


@lru_cache(maxsize=1)
def _load_cuda_nvfp4_mha():
    try:
        from ._cuda_ops import nvfp4_mha_fwd_sm120
        return nvfp4_mha_fwd_sm120
    except ImportError as e:
        raise ImportError(
            "CUDA operator 'nvfp4_mha_fwd_sm120' failed to load. "
            "Possible reasons: CUDA not available, or module not compiled."
        ) from e


def nvfp4_attention(q, k, v, out=None):
    _nvfp4_mha_sm120 = _load_cuda_nvfp4_mha()

    B, Nq, H, Dqk = q.shape
    _, _, _, Dv = v.shape

    q_fp4, q_sf, q_alpha = quantize(q, 1, 3)
    k_fp4, k_sf, k_alpha = quantize(k, 1, 3)
    v_fp4, v_sf, v_alpha = quantize(v, 3, 1, swizzle=True)

    if out is None:
        out = torch.empty((B, H, Nq, Dv), dtype=torch.bfloat16, device="cuda")

    _nvfp4_mha_sm120(out, q_fp4, q_sf, k_fp4, k_sf, v_fp4, v_sf,
                     q_alpha * k_alpha, v_alpha, Dqk)

    out = out.transpose(1, 2).contiguous()

    return out
