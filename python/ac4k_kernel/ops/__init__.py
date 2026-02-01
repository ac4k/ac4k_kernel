"""
AC4K Kernel Operators - Zero-overhead dispatch at import time.

All operators are compiled for the detected architecture at install time.
"""
from .dot_scale import dot_scale, nvfp4_dot_scale
from .quant import quantize, nvfp4_quantize, fp8_quantize, int8_quantize
from .attention import attention, nvfp4_mha_fwd, qk_int8_pv_fp8_mha_fwd
from .rope_3d import rope_3d_apply, rope_3d_apply_kernel

__all__ = [
    # High-level APIs
    "attention",
    "quantize",
    "rope_3d_apply",
    "dot_scale",
    # Direct kernel access (zero-overhead)
    "nvfp4_mha_fwd",
    "qk_int8_pv_fp8_mha_fwd",
    "nvfp4_quantize",
    "fp8_quantize",
    "int8_quantize",
    "nvfp4_dot_scale",
    "rope_3d_apply_kernel",
]
