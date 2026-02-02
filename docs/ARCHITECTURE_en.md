# AC4K Kernel Multi-Architecture Design

## Design Principles

**"Per-architecture optimization, zero runtime overhead"**

| Principle | Description |
|-----------|-------------|
| Architecture Isolation | Each architecture has fully independent code, no shared kernel implementations |
| Compile-time Dispatch | Hardware is detected at install time, only the target architecture is compiled |
| Zero-overhead Abstraction | Python API binds directly to C++ functions, no runtime branching |
| Minimal Dependencies | Only essential libraries, reducing build and runtime burden |

---

## Architecture Overview

```
                          +-----------------------------+
                          |      Python API Layer       |
                          |  ac4k_kernel.ops.*          |
                          |  (direct call, no dispatch) |
                          +-------------+---------------+
                                        |
                          +-------------v---------------+
                          |    _cuda_ops / _rocm_ops    |
                          |  (single backend at build)  |
                          +-------------+---------------+
                                        |
              +-------------------------+-------------------------+
              |                         |                         |
    +---------v---------+     +---------v---------+     +---------v---------+
    |   CUDA Backend    |     |   ROCm Backend    |     |  Future Backend   |
    |                   |     |                   |     |   (XLA/MPS/...)   |
    +-------------------+     +-------------------+     +-------------------+
    | sm120/ (RTX 5090) |     | gfx942/ (MI300X)  |
    | sm100/ (B200)     |     | gfx90a/ (MI250X)  |
    | sm90a/ (H100)     |     | ...               |
    | ...               |     |                   |
    +-------------------+     +-------------------+
```

---

## Directory Structure

```
ac4k_kernel/
├── include/ac4k_kernel/
│   ├── ops.h                    # Unified operator interface (backend/arch agnostic)
│   └── types.h                  # Common type definitions
│
├── lib/
│   ├── cuda/
│   │   ├── common/              # CUDA common utilities (traits, math, etc.)
│   │   │   ├── traits.cuh
│   │   │   ├── math.cuh
│   │   │   ├── dispatch.cuh
│   │   │   └── utils.cuh
│   │   ├── sm120/               # RTX 5090 specific
│   │   │   ├── mma.cuh          # namespace ac4k::sm120, MMA instructions
│   │   │   ├── register.cuh     # namespace ac4k::sm120, register management
│   │   │   ├── tma.cuh          # namespace ac4k::sm120, TMA
│   │   │   └── *.cu             # Implements public API in namespace ac4k
│   │   ├── sm100/               # B200/B100 specific (planned)
│   │   └── sm90a/               # H100/H200 specific (planned)
│   │
│   ├── rocm/
│   │   ├── common/              # ROCm common utilities
│   │   ├── gfx942/              # MI300X specific (planned)
│   │   └── gfx90a/              # MI250X specific (planned)
│   │
│   ├── cuda_bindings.cc         # CUDA pybind
│   └── rocm_bindings.cc         # ROCm pybind (planned)
│
├── python/ac4k_kernel/
│   ├── __init__.py
│   └── ops/
│       ├── attention.py
│       ├── quant.py
│       └── ...
│
└── setup.py                     # Compile-time architecture detection
```

---

## Compile-time Dispatch (Core Mechanism)

### Installation Flow

```
pip install .
     |
     v
+-----------------------------+
|  detect_backend()           |  Detect CUDA / ROCm
+-------------+---------------+
              |
     +--------+--------+
     v                 v
+---------+      +---------+
|  CUDA   |      |  ROCm   |
+----+----+      +----+----+
     |                |
     v                v
detect_arch()    detect_arch()
  sm120            gfx942
     |                |
     v                v
Build sm120/     Build gfx942/
     |                |
     v                v
_cuda_ops.so     _rocm_ops.so
```

### setup.py Core Logic

```python
def detect_backend():
    """Detect backend: CUDA or ROCm"""
    if os.environ.get('AC4K_BACKEND'):
        return os.environ['AC4K_BACKEND']

    # Detect CUDA
    if shutil.which('nvcc'):
        return 'cuda'

    # Detect ROCm
    if shutil.which('hipcc'):
        return 'rocm'

    raise RuntimeError("No supported backend found")


def detect_cuda_arch():
    """Detect CUDA architecture"""
    if arch := os.environ.get('AC4K_CUDA_ARCH'):
        return arch

    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    cap = result.stdout.strip().split('\n')[0]  # "12.0"
    major, minor = cap.split('.')
    return f"sm{major}{minor}"  # "sm120"


def detect_rocm_arch():
    """Detect ROCm architecture"""
    if arch := os.environ.get('AC4K_ROCM_ARCH'):
        return arch

    result = subprocess.run(['rocminfo'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'gfx' in line:
            return line.strip()  # "gfx942"
    return "gfx942"  # default


def get_extension():
    backend = detect_backend()

    if backend == 'cuda':
        arch = detect_cuda_arch()
        return CUDAExtension(
            name='ac4k_kernel._cuda_ops',
            sources=get_cuda_sources(arch),
            extra_compile_args={
                'nvcc': [f'-arch={arch}', '-O3', f'-DAC4K_ARCH_{arch.upper()}=1']
            }
        )

    elif backend == 'rocm':
        arch = detect_rocm_arch()
        return HIPExtension(
            name='ac4k_kernel._rocm_ops',
            sources=get_rocm_sources(arch),
            extra_compile_args=[f'--offload-arch={arch}', '-O3']
        )
```

### pybind Bindings (Compile-time Selection)

```cpp
// lib/cuda_bindings.cc
// Public API in namespace ac4k, single arch compiled at build time, no symbol conflicts
// Arch-specific helpers isolated via namespace ac4k::sm120 etc.

#include <pybind11/pybind11.h>
#include "ac4k_kernel/ops.h"

PYBIND11_MODULE(_cuda_ops, m) {
    // Direct binding to namespace ac4k functions, zero indirection
    m.def("mha_nvfp4_fwd", &ac4k::mha_nvfp4_fwd);
    m.def("mha_int8_x_fp8_fwd", &ac4k::mha_int8_x_fp8_fwd);
    m.def("quantize_nvfp4", &ac4k::quantize_nvfp4);
    m.def("quantize_fp8", &ac4k::quantize_fp8);
    // ...

    m.attr("__arch__") = kArchName;
    m.attr("__backend__") = "cuda";
}
```

### Python API (Zero Overhead)

```python
# python/ac4k_kernel/__init__.py

__version__ = "0.1.0"

# Direct import of compiled backend (no runtime branching)
try:
    from ._cuda_ops import __arch__, __backend__
    from ._cuda_ops import mha_nvfp4_fwd, mha_int8_x_fp8_fwd
    from ._cuda_ops import quantize_nvfp4, quantize_fp8, quantize_int8
    _backend = "cuda"
except ImportError:
    try:
        from ._rocm_ops import __arch__, __backend__
        _backend = "rocm"
    except ImportError:
        raise ImportError("No backend available. Install with CUDA or ROCm.")

def get_backend() -> str:
    return _backend

def get_arch() -> str:
    return __arch__
```

```python
# python/ac4k_kernel/ops/attention.py

from .._cuda_ops import mha_nvfp4_fwd, mha_int8_x_fp8_fwd


def attention(q, k, v, *, precision="nvfp4", **kwargs):
    """
    High-performance Attention

    precision selects kernel implementation, not architecture dispatch (zero overhead)
    """
    if precision == "nvfp4":
        return _nvfp4_attention(q, k, v, **kwargs)
    elif precision == "int8+fp8e4m3":
        return _int8_x_fp8_attention(q, k, v, **kwargs)
```

---

## Architecture Feature Comparison

### CUDA

| Architecture | GPU | NVFP4 | FP8 | TMA | Target Use Case |
|---|---|---|---|---|---|
| SM120 | RTX 5090 | ✅ | ✅ | ✅ | Consumer inference |
| SM100 | B200/B100 | ✅ | ✅ | ✅ | Datacenter training/inference |

### ROCm

| Architecture | GPU | FP8 | Matrix Core | Target Use Case |
|---|---|---|---|---|
| GFX942 | MI300X | ✅ | ✅ | Datacenter |
| GFX90a | MI250X | ✅ | ✅ | Datacenter |

---

## Code Sharing Strategy

### Not Shared (Architecture-specific)

- MMA/WMMA instruction wrappers
- TMA/LDS memory operations
- Kernel main loops
- Tile size configurations
- Register allocation

### Shared (in common/)

- Type traits (`traits.cuh`)
- Math functions (`math.cuh`)
- Error checking macros
- Python API wrappers

---

## Performance Guarantees

### Call Path Comparison

```
Traditional runtime dispatch:
  attention() -> get_backend() -> get_arch() -> dispatch() -> kernel
  Overhead: ~250ns/call

AC4K compile-time dispatch:
  attention() -> _cuda_ops.mha_nvfp4_fwd() -> kernel
  Overhead: ~100ns/call (Python function call itself)
```

### Transformer Inference Overhead

| Approach | Per-layer overhead (48 ops) | 24-layer model |
|---|---|---|
| Runtime dispatch | 12us | 288us |
| Compile-time dispatch | 4.8us | 115us |

---

## Build Options

```bash
# Auto-detect
pip install .

# Specify backend
AC4K_BACKEND=cuda pip install .
AC4K_BACKEND=rocm pip install .

# Specify architecture
AC4K_CUDA_ARCH=sm120 pip install .
AC4K_ROCM_ARCH=gfx90a pip install .

# Development mode
pip install -e . --no-build-isolation
```

### Build Acceleration

The build system automatically detects and enables the following optimizations:

| Optimization | Effect | How to enable |
|---|---|---|
| **Ninja** | File-level parallel compilation (replaces make) | `pip install ninja` |
| **ccache** | Caches build artifacts, speeds up recompilation | `apt install ccache` |
| **MAX_JOBS** | Controls parallel compilation jobs | `MAX_JOBS=N pip install ...` (default: half CPU cores) |
| **nvcc --threads** | nvcc intra-file parallelism (PTX->SASS) | Automatic |
| **-pipe** | Compiler uses pipes instead of temp files | Automatic |
| **Single-arch build** | Only compiles target GPU architecture, disables BuildExtension gencode injection | Automatic (via `-arch=sm_XXXa`) |

```bash
# First build
MAX_JOBS=$(nproc) pip install -e . --no-build-isolation

# Subsequent rebuilds (ccache hot, near-instant for unchanged files)
pip install -e . --no-build-isolation
```

### Environment Variables

| Variable | Description | Example |
|---|---|---|
| `AC4K_BACKEND` | Force backend | `cuda` / `rocm` |
| `AC4K_CUDA_ARCH` | Force CUDA architecture | `sm120` / `sm100` |
| `AC4K_ROCM_ARCH` | Force ROCm architecture | `gfx942` / `gfx90a` |
| `MAX_JOBS` | Parallel compilation jobs | `32` |

---

## Extending with New Architectures

### Adding a new CUDA architecture (e.g. SM130)

1. Create directory `lib/cuda/sm130/`
2. Implement kernel files
3. Add architecture detection in `setup.py`
4. Add `#elif` in `cuda_bindings.cc`

### Adding a new backend (e.g. XLA)

1. Create directory `lib/xla/`
2. Implement `xla_bindings.cc`
3. Add backend detection in `setup.py`
4. Add imports in Python `__init__.py`

---

## Operator Naming Convention

### C++ / pybind Layer: `{op}_{precision}[_{dir}]`

| Component | Description | Examples |
|-----------|-------------|----------|
| `{op}` | Operator type | `mha`, `quantize`, `gemm`, `rope3d` |
| `{precision}` | Data precision | `nvfp4`, `fp8`, `int8` |
| `{dir}` | Direction (optional) | `fwd`, `bwd` |

**Mixed precision**: When different stages use different precisions, separate with `_x_`:

```
mha_int8_x_fp8_fwd
     |      |    |
     op  QK stage  PV stage  direction
```

### Full Naming Map

| C++ / pybind Name | Python High-level API | Description |
|---|---|---|
| `mha_nvfp4_fwd` | `attention(precision="nvfp4")` | NVFP4 full-precision MHA |
| `mha_int8_x_fp8_fwd` | `attention(precision="int8+fp8e4m3")` | QK=INT8, PV=FP8 mixed-precision MHA |
| `quantize_nvfp4` | `quantize(precision="nvfp4")` | BF16 -> NVFP4 |
| `quantize_fp8` | `quantize(precision="fp8e4m3")` | BF16 -> FP8 |
| `quantize_int8` | `quantize(precision="int8")` | BF16 -> INT8 |
| `gemm_nvfp4` | `gemm()` | NVFP4 GEMM |
| `rope3d` | `rope3d()` | 3D RoPE (no precision/direction suffix) |

### Design Rationale

- **Op first**: Groups by operation type; IDE autocomplete shows all attention variants under `mha_`
- **Precision in the middle**: Describes "what precision to use", not "which architecture to run on"
- **`_x_` separator**: Marks precision of different compute stages in mixed-precision ops; more concise than `qk_int8_pv_fp8`
- **Python unified entry**: Users only need to remember `attention()`, `quantize()`, etc.; precision is passed as a parameter

---

## Operator List

| Operator | CUDA SM120 | CUDA SM100 |
|---|---|---|
| Attention (NVFP4) | ✅ | 📋 |
| Attention (FP8) | ✅ | 📋 |
| Attention (INT8) | ✅ | 📋 |
| Quantize (NVFP4) | ✅ | 📋 |
| Quantize (FP8) | ✅ | 📋 |
| Quantize (INT8) | ✅ | 📋 |
| RoPE 3D | ✅ | 📋 |
| Linear (NVFP4) | ✅ | 📋 |

✅ Implemented | 📋 Planned
