from functools import lru_cache


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


def nvfp4_attention(q, k, v, q_sf, k_sf, v_sf, alpha0, alpha1, out):
    _nvfp4_mha_sm120 = _load_cuda_nvfp4_mha()

    _nvfp4_mha_sm120(out, q, q_sf, k, k_sf, v, v_sf, alpha0, alpha1)

    return out
