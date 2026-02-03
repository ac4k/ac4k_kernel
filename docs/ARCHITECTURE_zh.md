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
                          │  (直接调用，无 dispatch)      │
                          └─────────────┬───────────────┘
                                        │
                          ┌─────────────▼───────────────┐
                          │    _cuda_ops / _rocm_ops    │
                          │    (编译时确定唯一后端)        │
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
│   ├── ops.h                    # 统一算子接口（backend/arch 无关）
│   └── types.h                  # 公共类型定义
│
├── lib/
│   ├── cuda/
│   │   ├── common/              # CUDA 通用工具 (traits, math, etc.)
│   │   │   ├── traits.cuh
│   │   │   ├── math.cuh
│   │   │   ├── dispatch.cuh
│   │   │   └── utils.cuh
│   │   ├── sm120/               # RTX 5090 专用
│   │   │   ├── mma.cuh          # namespace ac4k::sm120, MMA 指令
│   │   │   ├── register.cuh     # namespace ac4k::sm120, 寄存器管理
│   │   │   ├── tma.cuh          # namespace ac4k::sm120, TMA
│   │   │   └── *.cu             # 实现 namespace ac4k 中的公共 API
│   │   ├── sm100/               # B200/B100 专用（计划中）
│   │   └── sm90a/               # H100/H200 专用（计划中）
│   │
│   ├── rocm/
│   │   ├── common/              # ROCm 通用工具
│   │   ├── gfx942/              # MI300X 专用（计划中）
│   │   └── gfx90a/              # MI250X 专用（计划中）
│   │
│   ├── cuda_bindings.cc         # CUDA pybind
│   └── rocm_bindings.cc         # ROCm pybind（计划中）
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
// 公共 API 在 namespace ac4k 中，编译时只构建一个 arch，无符号冲突
// arch 内部 helpers 通过 namespace ac4k::sm120 等隔离

#include <pybind11/pybind11.h>
#include "ac4k_kernel/ops.h"

PYBIND11_MODULE(_cuda_ops, m) {
    // 直接绑定 namespace ac4k 中的函数，零间接层
    m.def("mha_nvfp4_fwd", &ac4k::mha_nvfp4_fwd);
    m.def("mha_int8_x_fp8_fwd", &ac4k::mha_int8_x_fp8_fwd);
    m.def("quantize_nvfp4", &ac4k::quantize_nvfp4);
    m.def("quantize_fp8", &ac4k::quantize_fp8);
    // ...

    m.attr("__arch__") = kArchName;
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
    高性能 Attention

    precision 选择 kernel 实现，不是架构分发（零开销）
    """
    if precision == "nvfp4":
        return _nvfp4_attention(q, k, v, **kwargs)
    elif precision == "int8+fp8e4m3":
        return _int8_x_fp8_attention(q, k, v, **kwargs)
```

---

## 架构特性对比

### CUDA

| 架构 | GPU | NVFP4 | FP8 | TMA | 目标场景 |
|------|-----|-------|-----|-----|---------|
| SM120 | RTX 5090 | ✅ | ✅ | ✅ | 消费级推理 |
| SM100 | B200/B100 | ✅ | ✅ | ✅ | 数据中心训练/推理 |

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
  attention() → _cuda_ops.mha_nvfp4_fwd() → kernel
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
AC4K_CUDA_ARCH=sm120 pip install .
AC4K_ROCM_ARCH=gfx90a pip install .

# 开发模式
pip install -e . --no-build-isolation
```

### 编译加速

构建系统自动检测并启用以下加速手段：

| 加速手段 | 作用 | 启用方式 |
|---------|------|---------|
| **Ninja** | 文件级并行编译（替代 make） | `pip install ninja` |
| **ccache** | 缓存编译产物，加速重编译 | `apt install ccache` |
| **MAX_JOBS** | 控制并行编译任务数 | `MAX_JOBS=N pip install ...`（默认：CPU 核数的一半） |
| **nvcc --threads** | nvcc 内部并行（PTX→SASS） | 自动启用 |
| **-pipe** | 编译器使用管道替代临时文件 | 自动启用 |
| **单架构编译** | 只编译目标 GPU 架构，禁用 BuildExtension 的 gencode 注入 | 自动（通过 `-arch=sm_XXXa`） |

```bash
# 首次编译
MAX_JOBS=$(nproc) pip install -e . --no-build-isolation

# 后续重编译（ccache 命中，未修改文件近乎瞬时）
pip install -e . --no-build-isolation
```

### 环境变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `AC4K_BACKEND` | 强制指定后端 | `cuda` / `rocm` |
| `AC4K_CUDA_ARCH` | 强制指定 CUDA 架构 | `sm120` / `sm100` |
| `AC4K_ROCM_ARCH` | 强制指定 ROCm 架构 | `gfx942` / `gfx90a` |
| `MAX_JOBS` | 并行编译任务数 | `32` |

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

## 算子命名规范

### C++ / pybind 层：`{op}_{precision}[_{dir}]`

| 组成 | 说明 | 示例 |
|------|------|------|
| `{op}` | 算子类型 | `mha`, `quantize`, `gemm`, `rope3d` |
| `{precision}` | 数据精度 | `nvfp4`, `fp8`, `int8` |
| `{dir}` | 方向（可选） | `fwd`, `bwd` |

**混合精度**：不同阶段使用不同精度时，用 `_x_` 分隔：

```
mha_int8_x_fp8_fwd
     │      │    │
     op  QK阶段  PV阶段  方向
```

### 完整命名映射

| C++ / pybind 名称 | Python 高级 API | 说明 |
|---|---|---|
| `mha_nvfp4_fwd` | `attention(precision="nvfp4")` | NVFP4 全精度 MHA |
| `mha_int8_x_fp8_fwd` | `attention(precision="int8+fp8e4m3")` | QK=INT8, PV=FP8 混合精度 MHA |
| `quantize_nvfp4` | `quantize(precision="nvfp4")` | BF16 → NVFP4 |
| `quantize_fp8` | `quantize(precision="fp8e4m3")` | BF16 → FP8 |
| `quantize_int8` | `quantize(precision="int8")` | BF16 → INT8 |
| `gemm_nvfp4` | `gemm()` | NVFP4 GEMM |
| `rope3d` | `rope3d()` | 3D RoPE（无精度/方向后缀） |

### 设计理由

- **op 在前**：按操作类型分组，IDE 自动补全时 `mha_` 列出所有 attention 变体
- **precision 在中**：描述"用什么精度做"，不是"在哪个架构上做"
- **`_x_` 分隔符**：混合精度时标注不同计算阶段的精度，比 `qk_int8_pv_fp8` 更简洁
- **Python 统一入口**：用户只需记住 `attention()`、`quantize()` 等高级 API，precision 作为参数传入

---

## 算子列表

| 算子 | CUDA SM120 | CUDA SM100 |
|------|------------|------------|
| Attention (NVFP4) | ✅ | 📋 |
| Attention (FP8) | ✅ | 📋 |
| Attention (INT8) | ✅ | 📋 |
| Quantize (NVFP4) | ✅ | 📋 |
| Quantize (FP8) | ✅ | 📋 |
| Quantize (INT8) | ✅ | 📋 |
| RoPE 3D | ✅ | 📋 |
| Linear (NVFP4) | ✅ | 📋 |

✅ 已实现 | 📋 计划中
