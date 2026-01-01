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


def _nvfp4_attention(q, k, v, layout="BNHD", out=None):
    assert layout in ["BNHD", "BHND"], "Unsupported layout: {}".format(layout)

    _nvfp4_mha_sm120 = _load_cuda_nvfp4_mha()

    B, Dqk = q.shape[0], q.shape[-1]
    Dv = v.shape[-1]
    if layout == "BNHD":
        Nq, H = q.shape[1], q.shape[2]
    else:
        H, Nq = q.shape[1], q.shape[2]

    N_dim = 1 if layout == "BNHD" else 2
    q_fp4, q_sf, q_alpha = quantize(q, N_dim, 3)
    k_fp4, k_sf, k_alpha = quantize(k, N_dim, 3)
    v_fp4, v_sf, v_alpha = quantize(v, 3, N_dim, swizzle=True)

    # Alloc output tensor
    if out is None:
        out_shape = (B, Nq, H, Dv) if layout == "BNHD" else (B, H, Nq, Dv)
        out = torch.empty(out_shape, dtype=torch.bfloat16, device="cuda")

    # Permute to BHND
    if layout == "BNHD":
        out = out.permute(0, 2, 1, 3)

    _nvfp4_mha_sm120(out, q_fp4, q_sf, q_alpha, k_fp4, k_sf, k_alpha, v_fp4,
                     v_sf, v_alpha, Dqk)

    # Permute back to origin layout
    if layout == "BNHD":
        out = out.permute(0, 2, 1, 3)

    return out


@lru_cache(maxsize=1)
def _load_cuda_qk_nvfp4_pv_fp8_mha():
    try:
        from ._cuda_ops import qk_nvfp4_pv_fp8_acc_fp16_mha_fwd_sm120
        return qk_nvfp4_pv_fp8_acc_fp16_mha_fwd_sm120
    except ImportError as e:
        raise ImportError(
            "CUDA operator 'qk_nvfp4_pv_fp8_acc_fp16_mha_fwd_sm120' failed to load. "
            "Possible reasons: CUDA not available, or module not compiled."
        ) from e


def _qk_nvfp4_pv_fp8_attention(q, k, v, layout="BNHD", out=None):
    assert layout in ["BNHD", "BHND"], "Unsupported layout: {}".format(layout)

    kernel = _load_cuda_qk_nvfp4_pv_fp8_mha()

    B, Dqk = q.shape[0], q.shape[-1]
    Dv = v.shape[-1]
    if layout == "BNHD":
        Nq, H = q.shape[1], q.shape[2]
    else:
        H, Nq = q.shape[1], q.shape[2]

    N_dim = 1 if layout == "BNHD" else 2
    k = k - k.mean(dim=N_dim, keepdim=True)
    q_fp4, q_sf, q_alpha = quantize(q, N_dim, 3)
    k_fp4, k_sf, k_alpha = quantize(k, N_dim, 3)
    v_fp4, v_sf = quantize(v, 3, N_dim, swizzle=True, precision="fp8e4m3")

    # Alloc output tensor
    if out is None:
        out_shape = (B, Nq, H, Dv) if layout == "BNHD" else (B, H, Nq, Dv)
        out = torch.empty(out_shape, dtype=torch.bfloat16, device="cuda")

    # Permute to BHND
    if layout == "BNHD":
        out = out.permute(0, 2, 1, 3)

    kernel(out, q_fp4, q_sf, q_alpha, k_fp4, k_sf, k_alpha, v_fp4, v_sf, Dqk)

    # Permute back to origin layout
    if layout == "BNHD":
        out = out.permute(0, 2, 1, 3)

    return out


@lru_cache(maxsize=1)
def _load_cuda_qk_int8_pv_fp8_mha():
    try:
        from ._cuda_ops import qk_int8_pv_fp8_mha_fwd_sm120
        return qk_int8_pv_fp8_mha_fwd_sm120
    except ImportError as e:
        raise ImportError(
            "CUDA operator 'qk_int8_pv_fp8_mha_fwd_sm120' failed to load. "
            "Possible reasons: CUDA not available, or module not compiled."
        ) from e


def _qk_int8_pv_fp8_attention(q, k, v, layout="BNHD", out=None):
    assert layout in ["BNHD", "BHND"], "Unsupported layout: {}".format(layout)

    kernel = _load_cuda_qk_int8_pv_fp8_mha()

    B, Dqk = q.shape[0], q.shape[-1]
    Dv = v.shape[-1]
    if layout == "BNHD":
        Nq, H = q.shape[1], q.shape[2]
    else:
        H, Nq = q.shape[1], q.shape[2]

    N_dim = 1 if layout == "BNHD" else 2
    k = k - k.mean(dim=N_dim, keepdim=True)
    q_fp4, q_sf = quantize(q, N_dim, 3, precision="int8")
    k_fp4, k_sf = quantize(k, N_dim, 3, precision="int8")
    v_fp4, v_sf = quantize(v, 3, N_dim, swizzle=True, precision="fp8e4m3")

    # Alloc output tensor
    if out is None:
        out_shape = (B, Nq, H, Dv) if layout == "BNHD" else (B, H, Nq, Dv)
        out = torch.empty(out_shape, dtype=torch.bfloat16, device="cuda")

    # Permute to BHND
    if layout == "BNHD":
        out = out.permute(0, 2, 1, 3)

    kernel(out, q_fp4, q_sf, k_fp4, k_sf, v_fp4, v_sf, Dqk)

    # Permute back to origin layout
    if layout == "BNHD":
        out = out.permute(0, 2, 1, 3)

    return out


def attention(q, k, v, layout="BNHD", precision="nvfp4", out=None):
    assert layout in ["BNHD", "BHND"], "Unsupported layout: {}".format(layout)
    assert precision in ["nvfp4", "nvfp4+fp8e4m3", "int8+fp8e4m3"
                         ], "Unsupported precision: {}".format(precision)

    if precision == "nvfp4":
        return _nvfp4_attention(q, k, v, layout, out)
    elif precision == "nvfp4+fp8e4m3":
        return _qk_nvfp4_pv_fp8_attention(q, k, v, layout, out)
    else:
        return _qk_int8_pv_fp8_attention(q, k, v, layout, out)
