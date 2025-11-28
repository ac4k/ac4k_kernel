from pathlib import Path
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import subprocess


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

# torch path
import torch

torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
torch_include_dir = os.path.join(os.path.dirname(torch.__file__), "include")

# pybind path
import pybind11

pybind11_include_dir = pybind11.get_include()

setup(name='ac4k_kernel',
      version='0.1.0',
      author="AC4K Team",
      description="High Performance Python Operations For NVFP4 type",
      packages=find_packages(where="python"),
      package_dir={"": "python"},
      ext_modules=[
          CUDAExtension(
              name='ac4k_kernel.ops._cuda_ops',
              sources=[
                  'lib/pybind_bindings.cc', 'lib/cuda/nvfp4_matmul_sm120.cu',
                  'lib/cuda/_internal_nvfp4_matmul_sm120.cu',
                  'lib/cuda/nvfp4_quant_sm120.cu'
              ],
              include_dirs=[
                  pybind11_include_dir,
                  torch_include_dir,
                  os.path.join(Path(__file__).resolve().parent, 'include'),
              ],
              extra_compile_args={
                  'cxx': ['-O3', '-std=c++17', '-fPIC'],
                  'nvcc': [
                      '-O3', '-std=c++17', '-arch=sm_120a', '--use_fast_math',
                      '--compiler-options'
                  ]
              },
              extra_link_args=[
                  "-lc10", "-lc10_cuda", "-ltorch", "-ltorch_cpu", "-lcuda",
                  "-lcudart", "-ltorch_cuda", f"-L{torch_lib_dir}",
                  f"-Wl,-rpath={torch_lib_dir}"
              ],
          )
      ],
      cmdclass={'build_ext': BuildExtension})
