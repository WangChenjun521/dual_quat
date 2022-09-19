from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="nn_cuda",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "nn_cuda",
            ["pytorch/nn_cuda_ops.cpp", "kernel/add2_kernel.cu","kernel/compute_distance.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)