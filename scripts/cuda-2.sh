#!/bin/bash

# module load CUDA/10.1.243-GCC-8.3.0

CC_OPTIONS="-O3 -Wall -Wextra -Wconversion -Wdouble-promotion \
    -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion -Wno-unknown-pragmas \
    -fopenmp -lm \
    -isystem src/lib $@"  # Pass any extra -D compiler flags

nvcc --compiler-options "$CC_OPTIONS" \
    -o cuda-2 \
    src/cuda-2.cu \

# srun --reservation=fri -G1 -n1 ./cuda-2
./cuda-2
