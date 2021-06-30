from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
  
# Get cutlass source directory
import os
filepath = os.path.dirname(os.path.abspath(__file__))
cutlass_path = filepath + "/../../../"

setup(
        name='pbatch_kernels',
        ext_modules=[
            CUDAExtension('pbatch_cuda', [
                'pbatch_interface.cpp', 
                'pbatch_wrapper.cu',
            ],
                          extra_compile_args={
                              'nvcc' : [
                                  '-I%s' % cutlass_path,
                                  '-arch=compute_75',
                                  '-code=compute_75',
                                  '-L/usr/local/cuda/lib64',
                                  '-lcuda',
                                  '-lcudart',
                                  '-lcublas',
                                  '-lcurand',
                                  '-U__CUDA_NO_HALF_CONVERSIONS__',
                                  '-U__CUDA_NO_HALF2_OPERATORS__',
                                  '-U__CUDA_NO_HALF_OPERATORS__'
                              ],
                              'cxx' : ['-Ofast',
                                       '-I%s' % cutlass_path]
                          }
            )
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
