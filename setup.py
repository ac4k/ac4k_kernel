from pathlib import Path
import os
from setuptools import setup, find_packages
import subprocess
import multiprocessing
import shutil
import pybind11
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the number of CPU cores
CPU_CORES = multiprocessing.cpu_count()
# Set the maximum number of parallel jobs to the number of CPU cores
MAX_PARALLEL_JOBS = min(CPU_CORES, 8)
os.environ['MAX_JOBS'] = str(MAX_PARALLEL_JOBS)


# Check for CUDA
def check_nvcc_version():
    try:
        result = subprocess.run(['nvcc', '--version'],
                                capture_output=True,
                                text=True)
        return result.stdout
    except:
        return "NVCC not found"


print("NVCC Version:")
print(check_nvcc_version())

# Check for Ninja to enable parallel compilation
ninja_path = shutil.which('ninja')
if ninja_path:
    print(f"Ninja found at {ninja_path}. Parallel compilation enabled.")
else:
    print(
        "Warning: Ninja not found. Install 'ninja' via 'pip install ninja' for faster parallel compilation. Falling back to serial mode."
    )

# torch path
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
torch_include_dir = os.path.join(os.path.dirname(torch.__file__), "include")

# pybind path
pybind11_include_dir = pybind11.get_include()

setup(
    name='ac4k_kernel',
    version='0.1.0',
    author="AC4K Team",
    description="High Performance Python Operations For NVFP4 type",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[
        CUDAExtension(
            name='ac4k_kernel.ops._cuda_ops',
            sources=[
                'lib/pybind_bindings.cc',
                # CUDA source files
                'lib/cuda/nvfp4_matmul_sm120.cu',
                'lib/cuda/_internal_nvfp4_matmul_sm120.cu',
                'lib/cuda/nvfp4_quant_sm120.cu',
                'lib/cuda/nvfp4_attention_sm120.cu',
                'lib/cuda/nvfp4_quantize_sm120.cu',
                'lib/cuda/rope_3d_apply.cu'
            ],
            include_dirs=[
                pybind11_include_dir,
                torch_include_dir,
                os.path.join(Path(__file__).resolve().parent, 'include'),
            ],
            extra_compile_args={
                'cxx': [
                    '-O3',
                    '-std=c++17',
                    '-fPIC',
                ],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '-arch=sm_120a',
                    '--use_fast_math',
                    '-DCUDA_COMPILE_CACHE=1',
                    '-w',
                    '--ptxas-options=-v',  # register usage report
                ]
            },
            extra_link_args=[
                "-lc10", "-lc10_cuda", "-ltorch", "-lcuda", "-lcudart",
                "-ltorch_cuda", f"-L{torch_lib_dir}",
                f"-Wl,-rpath={torch_lib_dir}"
            ],
        )
    ],
    cmdclass={
        'build_ext':
        BuildExtension.with_options(use_ninja=True, parallel=MAX_PARALLEL_JOBS)
    })
