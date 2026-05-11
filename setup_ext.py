from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path

csrc_path = Path(__file__).parent / "csrc"
sources = sorted(
    str(p) for p in csrc_path.glob("**/*") if p.suffix in [".cpp", ".cu"]
)

cpp_flags  = ["-O3", "-DWITH_CUDA"]
cuda_flags = ["-O3", "-DWITH_CUDA", "--use_fast_math"]

setup(
    name="vipe_ext_jit",
    ext_modules=[
        CUDAExtension(
            name="vipe_ext_jit",
            sources=sources,
            extra_compile_args={
                "cxx":  cpp_flags,
                "nvcc": cuda_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)