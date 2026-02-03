"""
Attention Operators - Zero Overhead Dispatch

Direct binding to compiled backend kernels.
"""
import math
import torch

from .quant import quantize

# Direct imports - no lazy loading, no runtime dispatch
from .._cuda_ops import mha_nvfp4_fwd, mha_int8_x_fp8_fwd


def _nvfp4_attention(q, k, v, layout, out=None):
    """NVFP4 precision attention implementation"""
    assert layout in ["BNHD", "BHND"], f"Unsupported layout: {layout}"

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
    mha_nvfp4_fwd(out, q_fp4, q_sf, q_alpha, k_fp4, k_sf, k_alpha,
                  v_fp4, v_sf, v_alpha, sm_scale)

    # Permute back to origin layout
    if layout == "BNHD":
        out = out.permute(0, 2, 1, 3)

    return out


def _int8_x_fp8_attention(q, k, v, layout, out=None):
    """INT8(QK) x FP8(PV) mixed-precision attention implementation"""
    assert layout in ["BNHD", "BHND"], f"Unsupported layout: {layout}"

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
    v_fp8, v_sf = quantize(v, 3, N_dim, swizzle=True, max_scale=2.25, precision="fp8e4m3")

    # Alloc output tensor
    if out is None:
        out_shape = (B, Nq, H, Dv) if layout == "BNHD" else (B, H, Nq, Dv)
        out = torch.empty(out_shape, dtype=torch.bfloat16, device="cuda")

    # Permute to BHND
    if layout == "BNHD":
        out = out.permute(0, 2, 1, 3)

    sm_scale = 1 / math.sqrt(Dqk)
    mha_int8_x_fp8_fwd(out, q_int8, q_sf, k_int8, k_sf, v_fp8, v_sf, sm_scale)

    # Permute back to origin layout
    if layout == "BNHD":
        out = out.permute(0, 2, 1, 3)

    return out


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    layout: str = "BNHD",
    precision: str = "int8+fp8e4m3",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """
    High-performance Multi-Head Attention

    Zero-overhead dispatch - kernel is selected at compile time.

    Args:
        q: Query tensor [B, N, H, D] or [B, H, N, D], bfloat16
        k: Key tensor [B, N, H, D] or [B, H, N, D], bfloat16
        v: Value tensor [B, N, H, D] or [B, H, N, D], bfloat16
        layout: "BNHD" or "BHND"
        precision: "nvfp4" or "int8+fp8e4m3"
        out: Optional pre-allocated output tensor

    Returns:
        Output tensor [B, N, H, D] or [B, H, N, D], bfloat16

    Note:
        - Head dimension D must be <= 128
        - Architecture-specific optimization is selected at install time
    """
    assert q.dtype == torch.bfloat16, "q dtype must be bfloat16"
    assert k.dtype == torch.bfloat16, "k dtype must be bfloat16"
    assert v.dtype == torch.bfloat16, "v dtype must be bfloat16"
    assert layout in ["BNHD", "BHND"], f"Unsupported layout: {layout}"
    assert precision in ["nvfp4", "int8+fp8e4m3"], f"Unsupported precision: {precision}"

    if out is not None:
        assert out.dtype == torch.bfloat16, "out dtype must be bfloat16"

    if precision == "nvfp4":
        return _nvfp4_attention(q, k, v, layout, out=out)
    else:
        return _int8_x_fp8_attention(q, k, v, layout, out=out)


# Direct kernel exports for maximum performance
__all__ = [
    "attention",
    "mha_nvfp4_fwd",
    "mha_int8_x_fp8_fwd",
]
