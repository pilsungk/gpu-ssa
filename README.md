# gpu-ssa
SSA on NVidia GPUs

First version of the implementation is based on my previous work like 7 years back.
Modified the old version to make it work with the latest CUDA environment (8.0). - replaced cutil with helper_cuda, etc.

This version is going to be the baseline for the later optimizated versions.

# build 
# need to specify CUDA_HOME/cuda-samples/Common for nvcc, for instance:
$ nvcc -c ssa_kernel.cu -I/home/pilsungk/cuda-samples/Common
