#!/bin/bash
# Clean project.

rm -rf build
rm -rf logs
rm -rf out
# rm -rf runs
find runs/ -name "*.png" -type f -delete

mkdir -p build  # compilation output
mkdir -p logs   # sbatch output
mkdir -p runs   # experiments output
mkdir -p out    # scratchpad output
