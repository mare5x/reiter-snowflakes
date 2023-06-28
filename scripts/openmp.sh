#!/bin/bash

threads=${1:-8}

gcc -O3 -Wall -Wextra -Wconversion -Wdouble-promotion \
    -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion -Wno-unknown-pragmas \
    -fopenmp \
    -lm \
    -isystem src/lib \
    -o openmp \
    src/openmp.c

# srun --reservation=fri --cpus-per-task=$threads --constraint=AMD perf stat -B -e cache-references,cache-misses,cycles,stalled-cycles-backend,instructions,branches,branch-misses ./openmp
./openmp
