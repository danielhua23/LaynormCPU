import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='my_cpu_extension', #在使用pip安装包或者在PyPI（Python Package Index）上发布包时会用到: pip install my_cpu_extension
    ext_modules=[
        CppExtension(
            'my_cpu_extension',# 实际python代码import的包名: import my_cpu_extension
            ['csrc/layernorm_cpu.cpp'],
            extra_compile_args=['-march=native']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=find_packages() # 找到所有的python包
)   


# setup(
#     name='lesson10',
#     ext_modules=[
#         cpp_extension.CUDAExtension(
#             name='lesson10._CUDA',
#             sources=[
#                 'csrc/layernorm.cu'
#             ],
#             include_dirs=['csrc/include'],
#             extra_link_args=['-lcudart'],
#             extra_compile_args={'cxx': ['-std=c++14', '-O3'],
#                                 'nvcc': ['-O3', '-std=c++14']},
#         ),
#     ],
#     cmdclass={
#         'build_ext': BuildExtension)
#     },
# )