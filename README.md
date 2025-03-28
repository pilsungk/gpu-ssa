# gpu-ssa
SSA on NVidia GPUs

First version of the implementation is based on my previous work like 7 years back.
Modified the old version to make it work with the latest CUDA environment (8.0). - replaced cutil with helper_cuda, etc.

This version is going to be the baseline for the later optimizated versions.

## build
Use the included Makefile to build the code. Need to change the 'cuda-samples' directory in the Makefile reflecting the installation environment before running make. 
