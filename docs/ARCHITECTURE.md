# AC4K Kernel 多架构设计

## 设计原则

**"每个架构独立优化，零运行时开销"**

| 原则 | 说明 |
|------|------|
| 架构隔离 | 不同架构代码完全独立，不共享 kernel 实现 |
| 编译时分发 | 安装时检测硬件，只编译目标架构 |
| 零开销抽象 | Python API 直接绑定到 C++ 函数，无运行时判断 |
| 最小依赖 | 只依赖必要的库，减少编译和运行时负担 |

---

## 架构总览

```
                          ┌─────────────────────────────┐
                          │      Python API Layer       │
                          │  ac4k_kernel.ops.*          │
                          │  (直接调用，无 dispatch)     │
                          └─────────────┬───────────────┘
                                        │
                          ┌─────────────▼───────────────┐
                          │    _cuda_ops / _rocm_ops    │
                          │    (编译时确定唯一后端)       │
                          └─────────────┬───────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
    ┌─────────▼─────────┐     ┌─────────▼─────────┐     ┌─────────▼─────────┐
    │   CUDA Backend    │     │   ROCm Backend    │     │  Future Backend   │
    │                   │     │                   │     │   (XLA/MPS/...)   │
    ├───────────────────┤     ├───────────────────┤     └───────────────────┘
    │ sm120/ (RTX 5090) │     │ gfx942/ (MI300X)  │
    │ sm100/ (B200)     │     │ gfx90a/ (MI250X)  │
    │ sm90a/ (H100)     │     │ ...               │
    │ ...               │     │                   │
    └───────────────────┘     └───────────────────┘
```

---

## 目录结构

```
ac4k_kernel/
├── include/ac4k_kernel/
│   ├── common/
│   │   └── types.h              # 公共类型定义
│   └── ops/
│       ├── cuda_ops.h           # CUDA 算子声明
│       └── rocm_ops.h           # ROCm 算子声明
│
├── lib/
│   ├── cuda/
│   │   ├── common/              # CUDA 通用工具 (traits, math, etc.)
│   │   │   ├── traits.cuh
│   │   │   ├── math.cuh
│   │   │   └── utils.cuh
│   │   ├── sm120/               # RTX 5090 专用
│   │   │   ├── mma.cuh          # SM120 MMA 指令
│   │   │   ├── tma.cuh          # SM120 TMA
│   │   │   ├── attention.cu
│   │   │   ├── quantize.cu
│   │   │   └── ...
│   │   ├── sm100/               # B200/B100 专用
│   │   └── sm90a/               # H100/H200 专用
│   │
│   ├── rocm/
│   │   ├── common/              # ROCm 通用工具
│   │   ├── gfx942/              # MI300X 专用
│   │   └── gfx90a/              # MI250X 专用
│   │
│   ├── cuda_bindings.cc         # CUDA pybind
│   └── rocm_bindings.cc         # ROCm pybind
│
├── python/ac4k_kernel/
│   ├── __init__.py
│   └── ops/
│       ├── attention.py
│       ├── quant.py
│       └── ...
│
└── setup.py                     # 编译时架构检测
```

---

## 编译时分发（核心机制）

### 安装流程

```
pip install .
     │
     ▼
┌─────────────────────────────┐
│  detect_backend()           │  检测 CUDA / ROCm
└─────────────┬───────────────┘
              │
     ┌────────┴────────┐
     ▼                 ▼
┌─────────┐      ┌─────────┐
│  CUDA   │      │  ROCm   │
└────┬────┘      └────┬────┘
     │                │
     ▼                ▼
detect_arch()    detect_arch()
  sm_120           gfx942
     │                │
     ▼                ▼
只编译 sm120/    只编译 gfx942/
     │                │
     ▼                ▼
_cuda_ops.so     _rocm_ops.so
```

### setup.py 核心逻辑

```python
def detect_backend():
    """检测后端：CUDA 或 ROCm"""
    if os.environ.get('AC4K_BACKEND'):
        return os.environ['AC4K_BACKEND']
    
    # 检测 CUDA
    if shutil.which('nvcc'):
        return 'cuda'
    
    # 检测 ROCm
    if shutil.which('hipcc'):
        return 'rocm'
    
    raise RuntimeError("No supported backend found")


def detect_cuda_arch():
    """检测 CUDA 架构"""
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
    """检测 ROCm 架构"""
    if arch := os.environ.get('AC4K_ROCM_ARCH'):
        return arch
    
    result = subprocess.run(['rocminfo'], capture_output=True, text=True)
    # 解析 gfx 架构
    for line in result.stdout.split('\n'):
        if 'gfx' in line:
            return line.strip()  # "gfx942"
    return "gfx942"  # 默认


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

### pybind 绑定（编译时选择）

```cpp
// lib/cuda_bindings.cc

#include <pybind11/pybind11.h>

// 编译时选择架构实现
#if defined(AC4K_ARCH_SM120)
  #include "cuda/sm120/ops.h"
  namespace impl = ac4k::cuda::sm120;
#elif defined(AC4K_ARCH_SM100)
  #include "cuda/sm100/ops.h"
  namespace impl = ac4k::cuda::sm100;
#elif defined(AC4K_ARCH_SM90A)
  #include "cuda/sm90a/ops.h"
  namespace impl = ac4k::cuda::sm90a;
#endif

PYBIND11_MODULE(_cuda_ops, m) {
    // 直接绑定，无间接层
    m.def("nvfp4_mha_fwd", &impl::nvfp4_mha_fwd);
    m.def("nvfp4_quantize", &impl::nvfp4_quantize);
    m.def("fp8_quantize", &impl::fp8_quantize);
    // ...
    
    m.attr("__arch__") = impl::kArchName;
    m.attr("__backend__") = "cuda";
}
```

### Python API（零开销）

```python
# python/ac4k_kernel/__init__.py

__version__ = "0.1.0"

# 直接导入编译好的后端（无运行时判断）
try:
    from ._cuda_ops import __arch__, __backend__
    from ._cuda_ops import nvfp4_mha_fwd, nvfp4_quantize, fp8_quantize
    _backend = "cuda"
except ImportError:
    try:
        from ._rocm_ops import __arch__, __backend__
        from ._rocm_ops import fp8_mha_fwd, fp8_quantize
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

import torch
from .. import _backend

if _backend == "cuda":
    from .._cuda_ops import nvfp4_mha_fwd, qk_int8_pv_fp8_mha_fwd
elif _backend == "rocm":
    from .._rocm_ops import fp8_mha_fwd


def attention(q, k, v, *, precision="nvfp4", **kwargs):
    """
    高性能 Attention
    
    precision 选择 kernel 实现，不是架构分发（零开销）
    """
    if _backend == "cuda":
        if precision == "nvfp4":
            return _nvfp4_attention(q, k, v, **kwargs)
        elif precision == "int8_fp8":
            return _int8_fp8_attention(q, k, v, **kwargs)
    elif _backend == "rocm":
        return _rocm_fp8_attention(q, k, v, **kwargs)
```

---

## 架构特性对比

### CUDA

| 架构 | GPU | NVFP4 | FP8 | TMA | 目标场景 |
|------|-----|-------|-----|-----|---------|
| SM120 | RTX 5090 | ✅ | ✅ | ✅ | 消费级推理 |
| SM100 | B200/B100 | ✅ | ✅ | ✅ | 数据中心训练/推理 |
| SM90a | H100/H200 | ❌ | ✅ | ✅ | 数据中心训练/推理 |
| SM89 | RTX 4090 | ❌ | ✅ | ❌ | 消费级推理 |

### ROCm

| 架构 | GPU | FP8 | Matrix Core | 目标场景 |
|------|-----|-----|-------------|---------|
| GFX942 | MI300X | ✅ | ✅ | 数据中心 |
| GFX90a | MI250X | ✅ | ✅ | 数据中心 |

---

## 代码共享策略

### 不共享（架构专用）

- MMA/WMMA 指令封装
- TMA/LDS 内存操作
- Kernel 主循环
- Tile 尺寸配置
- 寄存器分配

### 可共享（放在 common/）

- 类型萃取 (`traits.cuh`)
- 数学函数 (`math.cuh`)
- 错误检查宏
- Python API 封装

---

## 性能保证

### 调用链路对比

```
传统运行时分发:
  attention() → get_backend() → get_arch() → dispatch() → kernel
  开销: ~250ns/call

AC4K 编译时分发:
  attention() → _cuda_ops.nvfp4_mha_fwd() → kernel
  开销: ~100ns/call (Python 函数调用本身)
```

### Transformer 推理开销

| 方案 | 单层开销 (48 ops) | 24层模型 |
|------|------------------|---------|
| 运行时分发 | 12μs | 288μs |
| 编译时分发 | 4.8μs | 115μs |

---

## 构建选项

```bash
# 自动检测
pip install .

# 指定后端
AC4K_BACKEND=cuda pip install .
AC4K_BACKEND=rocm pip install .

# 指定架构
AC4K_CUDA_ARCH=sm100 pip install .
AC4K_ROCM_ARCH=gfx90a pip install .

# 开发模式
pip install -e . -v
```

---

## 扩展新架构

### 添加新 CUDA 架构 (如 SM130)

1. 创建目录 `lib/cuda/sm130/`
2. 实现 kernel 文件
3. 在 `setup.py` 添加架构检测
4. 在 `cuda_bindings.cc` 添加 `#elif`

### 添加新后端 (如 XLA)

1. 创建目录 `lib/xla/`
2. 实现 `xla_bindings.cc`
3. 在 `setup.py` 添加后端检测
4. 在 Python `__init__.py` 添加导入

---

## 算子列表

| 算子 | CUDA SM120 | CUDA SM90a | ROCm GFX942 |
|------|------------|------------|-------------|
| Attention (NVFP4) | ✅ | ❌ | ❌ |
| Attention (FP8) | ✅ | 📋 | 📋 |
| Attention (INT8) | ✅ | 📋 | 📋 |
| Quantize (NVFP4) | ✅ | ❌ | ❌ |
| Quantize (FP8) | ✅ | 📋 | 📋 |
| Quantize (INT8) | ✅ | 📋 | 📋 |
| RoPE 3D | ✅ | 📋 | 📋 |
| GEMM | ✅ | 📋 | 📋 |

✅ 已实现 | 📋 计划中 | ❌ 不支持
