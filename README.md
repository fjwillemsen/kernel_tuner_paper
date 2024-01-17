# Software repository for the upcoming Kernel Tuner paper


## Collection of Kernel Tuner backend performance
Backends tested: PyCUDA, CuPy, CUDA-Python, PyOpenCL. 
CUDA versions tested on: 11.2, 12.3.

When changing CUDA / OpenCL / driver versions, remember to check whether the packages and other dependencies should also be changed. 
In all cases, the latest package compatible with the CUDA version was used, namely: 
- For `PyCUDA`, `pycuda==2022.2.2`. Important: must be reinstalled when changing CUDA version as it compiles for the current CUDA version at install. 
- For CuPy, `cupy-cuda12x` with CUDA 12.3, and `cupy-cuda11x` with CUDA 11.2. 
- For CUDA-Python, only CUDA 12 was supported by `cuda-python==12.3.0` (`cuda-python==11.8.3` did not support Python 3.11). 
- For OpenCL, `PyOpenCL==2023.1.4` was used. 

## Collection of Searchspace generation performance
