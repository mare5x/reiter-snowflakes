#!/bin/bash
# Experiment for comparing all solutions.

module purge
module load CUDA/10.1.243-GCC-8.3.0

EXPERIMENT_NAME="all"

config='
-D ITERS=2000
-D ALPHA=1.0f
-D BETA=0.5f 
-D GAMMA=0.0001f
-D IMG_ITERS=200
-D IMG_W=2048
-D IMG_H=2048
-D PADDING=8
'

function run_serial () {
    local job_name="$EXPERIMENT_NAME-serial-$grid_n-$iter"
    local build_name="build/$job_name"
    gcc -O3 -Wall -Wextra -Wconversion -Wdouble-promotion \
        -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion -Wno-unknown-pragmas \
        -fopenmp \
        -lm \
        -isystem src/lib \
        -o "$build_name" \
        src/serial.c \
        $params
    sbatch -J "$job_name" << EOF
#!/bin/bash
#SBATCH --output=logs/%x.job_%j  # file name for stdout/stderr (%x will be replaced with the job name, %j with the jobid)
#SBATCH --time=2:00:00           # maximum wall time allocated for the job (D-H:MM:SS)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --reservation=fri
#SBATCH --constraint=AMD

srun "$build_name" > "$outdir/out.log"
EOF
}

function run_openmp () {
    local job_name="$EXPERIMENT_NAME-openmp-$grid_n-$iter"
    local build_name="build/$job_name"

    gcc -O3 -Wall -Wextra -Wconversion -Wdouble-promotion \
        -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion -Wno-unknown-pragmas \
        -fopenmp \
        -lm \
        -isystem src/lib \
        -o "$build_name" \
        src/openmp.c \
        $params
    sbatch -J "$job_name" << EOF
#!/bin/bash
#SBATCH --output=logs/%x.job_%j  # file name for stdout/stderr (%x will be replaced with the job name, %j with the jobid)
#SBATCH --time=0:30:00           # maximum wall time allocated for the job (D-H:MM:SS)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --reservation=fri
#SBATCH --constraint=AMD
#SBATCH --cpus-per-task=32

export OMP_NUM_THREADS=32
srun "$build_name" > "$outdir/out.log"
EOF
}

function run_cuda () {
    local CC_OPTIONS="-O3 -Wall -Wextra -Wconversion -Wdouble-promotion \
        -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion -Wno-unknown-pragmas \
        -fopenmp -lm \
        -isystem src/lib $params"
    local job_name="$EXPERIMENT_NAME-$name-$grid_n-$iter"
    local build_name="build/$job_name"
    nvcc --compiler-options "$CC_OPTIONS" \
        -o "$build_name" \
        "src/$name.cu"

    sbatch -J "$job_name" << EOF
#!/bin/bash
#SBATCH --output=logs/%x.job_%j  # file name for stdout/stderr (%x will be replaced with the job name, %j with the jobid)
#SBATCH --time=0:20:00           # maximum wall time allocated for the job (D-H:MM:SS)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=$([[ "$name" == "cuda-3" ]] && echo "2" || echo "1")
#SBATCH --reservation=fri

module purge
module load CUDA/10.1.243-GCC-8.3.0

srun "$build_name" > "$outdir/out.log"
EOF
}

for iter in {1..10}
do
    for grid_n in 3200
    do
        # Serial
        outdir="runs/$EXPERIMENT_NAME/serial/$grid_n/$iter"
        echo $outdir
        mkdir -p $outdir
        params="$(echo $config) -D ROWS=$grid_n -D COLS=$grid_n \
                                -D IMG_DIR=\"$outdir\""
        run_serial

        # OpenMP
        outdir="runs/$EXPERIMENT_NAME/openmp/$grid_n/$iter"
        echo $outdir
        mkdir -p $outdir
        params="$(echo $config) -D ROWS=$grid_n -D COLS=$grid_n \
                                -D IMG_DIR=\"$outdir\""
        run_openmp

        # CUDA
        for name in "cuda-1" "cuda-2" "cuda-3"
        do
            outdir="runs/$EXPERIMENT_NAME/$name/$grid_n/$iter"
            echo $outdir
            mkdir -p $outdir
            params="$(echo $config) -D ROWS=$grid_n -D COLS=$grid_n \
                                   '-D IMG_DIR=\"$outdir\"'"
            run_cuda
        done
    done
done
