from ac4k_kernel.ops import quantize

from functools import lru_cache
import math
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


def _nvfp4_attention(q, k, v, layout, out=None):
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

    sm_scale = 1 / math.sqrt(Dqk)
    _nvfp4_mha_sm120(out, q_fp4, q_sf, q_alpha, k_fp4, k_sf, k_alpha, v_fp4,
                     v_sf, v_alpha, sm_scale)

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


def _qk_int8_pv_fp8_attention(q, k, v, layout, out=None):
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
    q_int8, q_sf = quantize(q, N_dim, 3, precision="int8")
    k_int8, k_sf = quantize(k, N_dim, 3, precision="int8")
    v_fp8, v_sf = quantize(v,
                           3,
                           N_dim,
                           swizzle=True,
                           max_scale=2.25,
                           precision="fp8e4m3")

    # Alloc output tensor
    if out is None:
        out_shape = (B, Nq, H, Dv) if layout == "BNHD" else (B, H, Nq, Dv)
        out = torch.empty(out_shape, dtype=torch.bfloat16, device="cuda")

    # Permute to BHND
    if layout == "BNHD":
        out = out.permute(0, 2, 1, 3)

    sm_scale = 1 / math.sqrt(Dqk)
    kernel(out, q_int8, q_sf, k_int8, k_sf, v_fp8, v_sf, sm_scale)

    # Permute back to origin layout
    if layout == "BNHD":
        out = out.permute(0, 2, 1, 3)

    return out


class Ac4kAttentionOp(torch.nn.Module):

    def __init__(self):
        super(Ac4kAttentionOp, self).__init__()

    def forward(self, q, k, v, layout, precision, out=None):
        if precision == "nvfp4":
            return _nvfp4_attention(q, k, v, layout, out=out)
        else:
            return _qk_int8_pv_fp8_attention(q, k, v, layout, out=out)


def attention(q, k, v, layout="BNHD", precision="int8+fp8e4m3", out=None):
    """
    attention's Docstring

    Arguments:
        q: q tensor, only support bf16 dtype now.
        k: k tensor, only support bf16 dtype now.
        v: v tensor, only support bf16 dtype now.
        layout: BNHD or BHND, default BNHD. BNHD means (batch, seqlen, nheads, headdim), BHND means (batch, nheads, seqlen, headdim).
        precision: nvfp4 or int8+fp8e4m3, default int8+fp8e4m3. If set to nvfp4, q, k, v will be quantized to nvfp4 format, otherwise, q & k will be quantized to int8, v will be quantized to fp8e4m3 format.
        out: output tensor, if None, will create a new tensor. Only support bf16 dtype now.

        q, k, v headdim must be LE 128.

    Return:
        out: output tensor, only support bf16 dtype now.
    """

    assert q.dtype == torch.bfloat16, "q dtype must be bf16"
    assert k.dtype == torch.bfloat16, "k dtype must be bf16"
    assert v.dtype == torch.bfloat16, "v dtype must be bf16"
    assert layout in ["BNHD", "BHND"], "Unsupported layout: {}".format(layout)
    assert precision in ["nvfp4", "int8+fp8e4m3"
                         ], "Unsupported precision: {}".format(precision)
    if out:
        assert out.dtype == torch.bfloat16, "out dtype must be bf16"

    op = Ac4kAttentionOp()
    return op(q, k, v, layout, precision, out=out)
