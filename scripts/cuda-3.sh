#!/bin/bash

# module load CUDA/10.1.243-GCC-8.3.0

CC_OPTIONS="-O3 -Wall -Wextra -Wconversion -Wdouble-promotion \
    -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion -Wno-unknown-pragmas \
    -fopenmp -lm \
    -isystem src/lib $@"  # Pass any extra -D compiler flags

nvcc --compiler-options "$CC_OPTIONS" \
    -o cuda-3 \
    src/cuda-3.cu \

# srun --reservation=fri -G2 -n1 ./cuda-3
./cuda-3
