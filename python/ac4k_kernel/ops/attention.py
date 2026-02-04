"""
Attention Operators - Zero Overhead Dispatch

Direct binding to compiled backend kernels.
"""
import math
import torch
import triton
import triton.language as tl

from .quant import quantize
from .utils import ceil_div, align_up

# Direct imports - no lazy loading, no runtime dispatch
from .._cuda_ops import mha_nvfp4_fwd, mha_int8_x_fp8_fwd


@triton.jit
def _triton_block_token_quantize(x_ptr, x_b_stride, x_h_stride, x_n_stride,
                                 x_d_stride, y_ptr, y_b_stride, y_h_stride,
                                 y_n_stride, y_d_stride, scale_ptr,
                                 scale_b_stride, scale_h_stride,
                                 scale_n_stride, n, d, do,
                                 BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    x_ptrs = x_ptr + pid_b * x_b_stride + pid_h * x_h_stride + offs_n[:, None] * x_n_stride + offs_d[
        None, :] * x_d_stride
    x = tl.load(x_ptrs, mask=(offs_n[:, None] < n) & (offs_d[None, :] < d))
    scale = tl.max(tl.abs(x)).to(tl.float32) / 127

    if scale == 0:
        y = tl.full([BLOCK_N, BLOCK_D], 0, dtype=tl.int8)
    else:
        y = tl.cast(x.to(tl.float32) / scale, dtype=tl.int8)

    tl.store(
        scale_ptr + pid_b * scale_b_stride + pid_h * scale_h_stride +
        pid_n * scale_n_stride, scale)

    y_ptrs = y_ptr + pid_b * y_b_stride + pid_h * y_h_stride + offs_n[:, None] * y_n_stride + offs_d[
        None, :] * y_d_stride
    tl.store(y_ptrs, y, mask=(offs_n[:, None] < n) & (offs_d[None, :] < do))


def _int8_quantize(x, layout, dim, type="token", BLOCK_SIZE=64, HDIM_ALIGN=16):
    assert HDIM_ALIGN % 16 == 0, "HDIM_ALIGN must be a multiple of 16"
    assert type == "token", "Only support token quantization yet."
    assert BLOCK_SIZE >= 16
    assert (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0
    assert layout in ["BNHD", "BHND"], "Unsupported layout: {}".format(layout)
    if layout == "BNHD":
        B, N, H, D = x.shape
        b_stride, n_stride, h_stride, d_stride = x.stride(0), x.stride(
            1), x.stride(2), x.stride(3)
        assert dim in [1, 3]
    else:
        B, H, N, D = x.shape
        b_stride, h_stride, n_stride, d_stride = x.stride(0), x.stride(
            1), x.stride(2), x.stride(3)
        assert dim in [2, 3]

    scale = torch.empty([B, H, ceil_div(x.shape[dim], BLOCK_SIZE)],
                        device=x.device,
                        dtype=torch.float32)
    y = torch.empty([B, H, N, align_up(D, HDIM_ALIGN)],
                    device=x.device,
                    dtype=torch.int8)

    BLOCK_N = BLOCK_SIZE
    BLOCK_D = triton.next_power_of_2(y.shape[-1])
    grid = (scale.shape[-1], H, B)

    _triton_block_token_quantize[grid](x,
                                       b_stride,
                                       h_stride,
                                       n_stride,
                                       d_stride,
                                       y,
                                       y.stride(0),
                                       y.stride(1),
                                       y.stride(2),
                                       y.stride(3),
                                       scale,
                                       scale.stride(0),
                                       scale.stride(1),
                                       scale.stride(2),
                                       N,
                                       D,
                                       y.shape[-1],
                                       BLOCK_N=BLOCK_N,
                                       BLOCK_D=BLOCK_D)

    return y, scale


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
    mha_nvfp4_fwd(out, q_fp4, q_sf, q_alpha, k_fp4, k_sf, k_alpha, v_fp4, v_sf,
                  v_alpha, sm_scale)

    # Permute back to origin layout
    if layout == "BNHD":
        out = out.permute(0, 2, 1, 3)

    return out


def _int8_x_fp8_attention(q,
                          k,
                          v,
                          layout,
                          Q_QUANTIZE_BLOCK_SIZE=64,
                          K_QUANTIZE_BLOCK_SIZE=128,
                          out=None):
    """INT8(QK) x FP8(PV) mixed-precision attention implementation"""
    assert layout in ["BNHD", "BHND"], f"Unsupported layout: {layout}"

    B, Dqk = q.shape[0], q.shape[-1]
    Dv = v.shape[-1]
    if layout == "BNHD":
        Nq, H = q.shape[1], q.shape[2]
    else:
        H, Nq = q.shape[1], q.shape[2]

    N_dim = 1 if layout == "BNHD" else 2

    # quantize Q
    k = k - k.mean(dim=N_dim, keepdim=True)
    q_int8, q_sf = _int8_quantize(q,
                                  layout,
                                  N_dim,
                                  BLOCK_SIZE=Q_QUANTIZE_BLOCK_SIZE)
    # quantize K
    k_int8, k_sf = _int8_quantize(k,
                                  layout,
                                  N_dim,
                                  BLOCK_SIZE=K_QUANTIZE_BLOCK_SIZE)
    # quantize V
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
    mha_int8_x_fp8_fwd(out, q_int8, q_sf, Q_QUANTIZE_BLOCK_SIZE, k_int8, k_sf,
                       K_QUANTIZE_BLOCK_SIZE, v_fp8, v_sf, sm_scale)

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
    assert precision in ["nvfp4",
                         "int8+fp8e4m3"], f"Unsupported precision: {precision}"

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
