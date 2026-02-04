"""AC4K Kernel Operators - Zero-overhead dispatch at import time.

All operators are compiled for the detected architecture at install time.
"""
from .linear import linear, linear_nvfp4
from .quant import quantize, quantize_nvfp4, quantize_fp8, quantize_int8
from .attention import attention, mha_nvfp4_fwd, mha_int8_x_fp8_fwd
from .sparse_linear_attention import SparseLinearAttention
from .rope_3d import rope3d, rope3d_kernel

__all__ = [
    # High-level APIs
    "attention",
    "SparseLinearAttention",
    "quantize",
    "linear",
    "rope3d",
    # Direct kernel access (zero-overhead)
    "mha_nvfp4_fwd",
    "mha_int8_x_fp8_fwd",
    "quantize_nvfp4",
    "quantize_fp8",
    "quantize_int8",
    "linear_nvfp4",
    "rope3d_kernel",
]
