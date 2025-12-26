module load conda
conda activate ~/.conda/envs/pytorch/
torchrun --nnodes=1 --nproc-per-node=1 demo_gpu.py
echo "<----------------------------------->"
torchrun --nnodes=1 --nproc-per-node=2 demo_gpu.py
