#!/usr/bin/env bash
module load conda
conda activate ~/.conda/envs/pytorch/
# Default number of processes per node (can be overridden by exporting NPROC_PER_NODE before running)
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

# Script to run: can be passed as first argument or via env var SCRIPT
# Usage: ./run_gpu.sh [script_to_run] [-- optional args passed to the script]
# SCRIPT=${1:-${SCRIPT:-demo.py}}
SCRIPT="cpu.py"
# remaining args (if any) are passed to the executed script
# SCRIPT_ARGS="${@:2}"
SCRIPT_ARGS=""

# echo "Running script: $SCRIPT $SCRIPT_ARGS with NPROC_PER_NODE=$NPROC_PER_NODE"
# rm -f results/results_cpu.csv
# echo "size;batch_size;total_loading_time;total_computing_time;total_time\n" >> results/results_cpu.csv



for batch_size in 16 32 64 128 256
do
    # torchrun --nnodes=1 --nproc-per-node=1 "$SCRIPT" "$batch_size"
    # torchrun --nnodes=1 --nproc-per-node=1 "$SCRIPT" "$batch_size"
    # torchrun --nnodes=1 --nproc-per-node=1 "$SCRIPT" "$batch_size"

    # torchrun --nnodes=1 --nproc-per-node=2 "$SCRIPT" "$batch_size"
    # torchrun --nnodes=1 --nproc-per-node=2 "$SCRIPT" "$batch_size"
    # torchrun --nnodes=1 --nproc-per-node=2 "$SCRIPT" "$batch_size"

    # torchrun --nnodes=1 --nproc-per-node=3 "$SCRIPT" "$batch_size"
    # torchrun --nnodes=1 --nproc-per-node=3 "$SCRIPT" "$batch_size"
    # torchrun --nnodes=1 --nproc-per-node=3 "$SCRIPT" "$batch_size"

    # torchrun --nnodes=1 --nproc-per-node=4 "$SCRIPT" "$batch_size"
    # torchrun --nnodes=1 --nproc-per-node=4 "$SCRIPT" "$batch_size"
    # torchrun --nnodes=1 --nproc-per-node=4 "$SCRIPT" "$batch_size"

    # torchrun --nnodes=1 --nproc-per-node=5 "$SCRIPT" "$batch_size"
    # torchrun --nnodes=1 --nproc-per-node=5 "$SCRIPT" "$batch_size"
    # torchrun --nnodes=1 --nproc-per-node=5 "$SCRIPT" "$batch_size"

    # torchrun --nnodes=1 --nproc-per-node=6 "$SCRIPT" "$batch_size"
    # torchrun --nnodes=1 --nproc-per-node=6 "$SCRIPT" "$batch_size"
    # torchrun --nnodes=1 --nproc-per-node=6 "$SCRIPT" "$batch_size"

    torchrun --nnodes=1 --nproc-per-node=7 "$SCRIPT" "$batch_size"
    torchrun --nnodes=1 --nproc-per-node=7 "$SCRIPT" "$batch_size"
    torchrun --nnodes=1 --nproc-per-node=7 "$SCRIPT" "$batch_size"

    torchrun --nnodes=1 --nproc-per-node=8 "$SCRIPT" "$batch_size"
    torchrun --nnodes=1 --nproc-per-node=8 "$SCRIPT" "$batch_size"
    torchrun --nnodes=1 --nproc-per-node=8 "$SCRIPT" "$batch_size"
done