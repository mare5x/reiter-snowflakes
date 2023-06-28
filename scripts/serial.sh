#!/bin/bash

gcc -O3 -Wall -Wextra -Wconversion -Wdouble-promotion \
    -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion -Wno-unknown-pragmas \
    -fopenmp \
    -lm \
    -isystem src/lib \
    -o serial \
    src/serial.c \
    "$@"  # Pass any extra -D compiler flags

# srun --reservation=fri ./serial
./serial
