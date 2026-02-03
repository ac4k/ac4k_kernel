"""
AC4K Kernel - High Performance GPU Operators

Architecture-specific optimizations with zero-overhead dispatch.
Backend and architecture are determined at install time.

Usage:
    import ac4k_kernel
    print(ac4k_kernel.get_backend())  # 'cuda' or 'rocm'
    print(ac4k_kernel.get_arch())     # 'sm120', 'gfx942', etc.
"""

__version__ = "0.1.0"

# Backend detection - determined at compile time, no runtime overhead
_backend = None
_arch = None

try:
    from ._cuda_ops import (
        __arch__,
        __backend__,
        # Attention
        mha_nvfp4_fwd,
        mha_int8_x_fp8_fwd,
        # Quantize
        quantize_nvfp4,
        quantize_fp8,
        quantize_int8,
        # Linear
        linear_nvfp4,
        # RoPE
        rope3d,
    )
    _backend = __backend__
    _arch = __arch__

except ImportError:
    try:
        from ._rocm_ops import (
            __arch__,
            __backend__,
            # ROCm operators (when available)
        )
        _backend = __backend__
        _arch = __arch__
    except ImportError:
        pass

if _backend is None:
    raise ImportError("AC4K Kernel: No backend available.\n"
                      "Please install with CUDA or ROCm support:\n"
                      "  pip install . (auto-detect)\n"
                      "  AC4K_BACKEND=cuda pip install .\n"
                      "  AC4K_BACKEND=rocm pip install .")


def get_backend() -> str | None:
    """Return the compiled backend ('cuda' or 'rocm')"""
    return _backend


def get_arch() -> str | None:
    """Return the compiled architecture (e.g., 'sm120', 'gfx942')"""
    return _arch


def get_info() -> dict:
    """Return detailed build information"""
    return {
        "version": __version__,
        "backend": _backend,
        "arch": _arch,
    }


# Public API
__all__ = [
    # Meta
    "__version__",
    "get_backend",
    "get_arch",
    "get_info",
    # Operators (if CUDA)
    "mha_nvfp4_fwd",
    "mha_int8_x_fp8_fwd",
    "quantize_nvfp4",
    "quantize_fp8",
    "quantize_int8",
    "linear_nvfp4",
    "rope3d",
]
