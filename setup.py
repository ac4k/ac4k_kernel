"""
AC4K Kernel Build Configuration

Supports compile-time backend and architecture selection for zero-overhead dispatch.

Environment Variables:
    AC4K_BACKEND: Force backend (cuda/rocm)
    AC4K_CUDA_ARCH: Force CUDA architecture (sm120/sm100)
    AC4K_ROCM_ARCH: Force ROCm architecture (gfx942/gfx90a)
    MAX_JOBS: Override parallel compilation jobs (default: CPU cores)
"""
from pathlib import Path
import os
import glob
from setuptools import setup, find_packages
import subprocess
import multiprocessing
import shutil
import pybind11
import torch

#===----------------------------------------------------------------------===//
# Build Acceleration
#===----------------------------------------------------------------------===//

# Parallel compilation: respect MAX_JOBS env var, default to half CPU cores
# to avoid starving the system (SSH, IDE, etc.)
CPU_CORES = multiprocessing.cpu_count()
MAX_PARALLEL_JOBS = int(os.environ.get('MAX_JOBS', max(1, CPU_CORES // 2)))
os.environ['MAX_JOBS'] = str(MAX_PARALLEL_JOBS)

# Paths
ROOT_DIR = Path(__file__).resolve().parent
INCLUDE_DIR = ROOT_DIR / 'include'
LIB_DIR = ROOT_DIR / 'lib'

torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
torch_include_dir = os.path.join(os.path.dirname(torch.__file__), "include")
pybind11_include_dir = pybind11.get_include()

#===----------------------------------------------------------------------===//
# Backend Detection
#===----------------------------------------------------------------------===//


def detect_backend() -> str:
    """Detect available backend: cuda or rocm"""
    # Environment override
    if backend := os.environ.get('AC4K_BACKEND'):
        return backend.lower()

    # Auto-detect CUDA
    if shutil.which('nvcc'):
        return 'cuda'

    # Auto-detect ROCm
    if shutil.which('hipcc'):
        return 'rocm'

    raise RuntimeError("No supported backend found. Install CUDA or ROCm.\n"
                       "Or set AC4K_BACKEND=cuda/rocm to force a backend.")


#===----------------------------------------------------------------------===//
# CUDA Architecture Detection
#===----------------------------------------------------------------------===//

CUDA_ARCH_MAP = {
    '12.0': 'sm120',
    '10.0': 'sm100',
}

SUPPORTED_CUDA_ARCHS = ['sm120', 'sm100']


def detect_cuda_arch() -> str:
    """Detect CUDA architecture from GPU"""
    # Environment override
    if arch := os.environ.get('AC4K_CUDA_ARCH'):
        arch = arch.lower().replace('_', '')
        if arch not in SUPPORTED_CUDA_ARCHS:
            available = [
                d.name for d in (LIB_DIR / 'cuda').iterdir()
                if d.is_dir() and d.name.startswith('sm')
            ]
            raise ValueError(
                f"Unsupported CUDA arch: {arch}. Available: {available}")
        return arch

    # Try nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5)
        if result.returncode == 0:
            cap = result.stdout.strip().split('\n')[0]  # e.g., "12.0"
            if cap in CUDA_ARCH_MAP:
                return CUDA_ARCH_MAP[cap]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Try torch.cuda
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            cap = f"{props.major}.{props.minor}"
            if cap in CUDA_ARCH_MAP:
                return CUDA_ARCH_MAP[cap]
    except:
        pass

    # Fail instead of silent default - wrong arch causes runtime errors
    raise RuntimeError("[ac4k] Could not auto-detect CUDA architecture.\n"
                       f"[ac4k] Supported: {SUPPORTED_CUDA_ARCHS}\n"
                       "[ac4k] Please set AC4K_CUDA_ARCH, e.g.:\n"
                       "[ac4k]   AC4K_CUDA_ARCH=sm120 pip install .")


def get_cuda_extension(arch: str):
    """Build CUDAExtension for specified architecture"""
    from torch.utils.cpp_extension import CUDAExtension

    arch_dir = LIB_DIR / 'cuda' / arch
    common_dir = LIB_DIR / 'cuda' / 'common'

    if not arch_dir.exists():
        available = [
            d.name for d in (LIB_DIR / 'cuda').iterdir()
            if d.is_dir() and d.name.startswith('sm')
        ]
        raise RuntimeError(
            f"No implementation for {arch}. Available: {available}")

    # Collect source files
    sources = [str(LIB_DIR / 'cuda_bindings.cc')]
    sources += glob.glob(str(arch_dir / '*.cu'))

    # Arch flag for nvcc - explicit mapping, no fragile fallback
    NVCC_ARCH_FLAGS = {
        'sm120': '-arch=sm_120a',
        'sm100': '-arch=sm_100a',
    }
    if arch not in NVCC_ARCH_FLAGS:
        raise ValueError(
            f"No nvcc arch flag defined for {arch}. Add it to NVCC_ARCH_FLAGS."
        )
    arch_flag = NVCC_ARCH_FLAGS[arch]

    # nvcc intra-file parallelism (CUDA 11.2+)
    nvcc_threads = min(MAX_PARALLEL_JOBS, 4)

    print(f"[ac4k] Building CUDA extension for {arch}")
    print(f"[ac4k] Sources: {len(sources)} files")
    print(
        f"[ac4k] Parallel jobs: {MAX_PARALLEL_JOBS}, nvcc threads: {nvcc_threads}"
    )

    return CUDAExtension(
        name='ac4k_kernel._cuda_ops',
        sources=sources,
        include_dirs=[
            pybind11_include_dir,
            torch_include_dir,
            str(INCLUDE_DIR),
            str(common_dir),
            str(arch_dir),
        ],
        extra_compile_args={
            'cxx': [
                '-O3',
                '-std=c++17',
                '-fPIC',
                '-DNDEBUG',
                '-pipe',  # Use pipes instead of temp files
            ],
            'nvcc': [
                '-O3',
                '-std=c++17',
                arch_flag,
                '--use_fast_math',
                f'-DAC4K_ARCH_{arch.upper()}=1',
                f'-DAC4K_BACKEND_CUDA=1',
                '-DNDEBUG',
                '-w',
                '--expt-relaxed-constexpr',
                f'--threads={nvcc_threads}',  # Intra-file parallelism
                '-Xcompiler',
                '-pipe',  # Host compiler uses pipes
                '--ptxas-options=-v',  # register usage report
            ]
        },
        extra_link_args=[
            "-lc10", "-lc10_cuda", "-ltorch", "-lcuda", "-lcudart",
            "-ltorch_cuda", f"-L{torch_lib_dir}", f"-Wl,-rpath={torch_lib_dir}"
        ],
    )


#===----------------------------------------------------------------------===//
# ROCm Architecture Detection
#===----------------------------------------------------------------------===//

SUPPORTED_ROCM_ARCHS = ['gfx942', 'gfx90a']


def detect_rocm_arch() -> str:
    """Detect ROCm architecture from GPU"""
    # Environment override
    if arch := os.environ.get('AC4K_ROCM_ARCH'):
        arch = arch.lower()
        if arch not in SUPPORTED_ROCM_ARCHS:
            raise ValueError(
                f"Unsupported ROCm arch: {arch}. Supported: {SUPPORTED_ROCM_ARCHS}"
            )
        return arch

    # Try rocminfo
    try:
        result = subprocess.run(['rocminfo'],
                                capture_output=True,
                                text=True,
                                timeout=10)
        for line in result.stdout.split('\n'):
            for supported in SUPPORTED_ROCM_ARCHS:
                if supported in line.lower():
                    return supported
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Default
    print("[ac4k] Warning: Could not detect ROCm arch, defaulting to gfx942")
    return 'gfx942'


def get_rocm_extension(arch: str):
    """Build HIPExtension for specified architecture"""
    from torch.utils.cpp_extension import HIPExtension

    arch_dir = LIB_DIR / 'rocm' / arch
    common_dir = LIB_DIR / 'rocm' / 'common'

    if not arch_dir.exists():
        raise RuntimeError(
            f"No implementation for {arch}. ROCm support not yet available.")

    sources = [str(LIB_DIR / 'rocm_bindings.cc')]
    sources += glob.glob(str(arch_dir / '*.cpp'))

    print(f"[ac4k] Building ROCm extension for {arch}")

    return HIPExtension(
        name='ac4k_kernel._rocm_ops',
        sources=sources,
        include_dirs=[
            pybind11_include_dir,
            str(INCLUDE_DIR),
            str(common_dir),
            str(arch_dir),
        ],
        extra_compile_args=[
            '-O3',
            '-std=c++17',
            f'--offload-arch={arch}',
            f'-DAC4K_ARCH_{arch.upper()}=1',
            f'-DAC4K_BACKEND_ROCM=1',
        ],
    )


#===----------------------------------------------------------------------===//
# Main Setup
#===----------------------------------------------------------------------===//


def get_extensions():
    """Get extensions based on detected backend"""
    backend = detect_backend()
    print(f"[ac4k] Detected backend: {backend}")

    if backend == 'cuda':
        arch = detect_cuda_arch()
        print(f"[ac4k] Detected CUDA architecture: {arch}")
        return [get_cuda_extension(arch)]

    elif backend == 'rocm':
        arch = detect_rocm_arch()
        print(f"[ac4k] Detected ROCm architecture: {arch}")
        return [get_rocm_extension(arch)]

    else:
        raise RuntimeError(f"Unknown backend: {backend}")


# Check for Ninja - required for parallel file-level compilation
USE_NINJA = bool(shutil.which('ninja'))
if USE_NINJA:
    print(f"[ac4k] Ninja found, parallel file compilation enabled")
else:
    print(
        "[ac4k] Warning: Ninja not found. Install via 'pip install ninja' for faster builds."
    )

# Print build configuration summary
print(f"[ac4k] MAX_JOBS={MAX_PARALLEL_JOBS}, CPU cores={CPU_CORES}")

from torch.utils.cpp_extension import BuildExtension

# Prevent BuildExtension from adding -gencode flags.
# We use -arch=sm_120a (with 'a' suffix for accelerated features like TCGEN05).
# BuildExtension's TORCH_CUDA_ARCH_LIST only generates -gencode compute_XX/sm_XX
# (without 'a'), which would cause nvcc to also compile for the non-accelerated
# target — failing on sm_120a-only instructions in mma.cuh.
import torch.utils.cpp_extension as _cpp_ext

_cpp_ext._get_cuda_arch_flags = lambda *args, **kwargs: []

setup(name='ac4k_kernel',
      version='0.1.0',
      author="AC4K Team",
      description=
      "High Performance GPU Operators with Architecture-Specific Optimization",
      packages=find_packages(where="python"),
      package_dir={"": "python"},
      ext_modules=get_extensions(),
      python_requires=">=3.10",
      install_requires=["torch>=2.0", "triton>=2.0"],
      cmdclass={
          'build_ext':
          BuildExtension.with_options(
              use_ninja=USE_NINJA,
              parallel=MAX_PARALLEL_JOBS,
          )
      })
