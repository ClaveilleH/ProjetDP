module load conda
conda activate ~/.conda/envs/pytorch/
# Default number of processes per node (can be overridden by exporting NPROC_PER_NODE before running)
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

# torchrun --nnodes=1 --nproc-per-node=${NPROC_PER_NODE} demo_gpu.py
# echo "<----------------------------------->"
# torchrun --nnodes=1 --nproc-per-node=2 demo_gpu.py

torchrun --nnodes=1 --nproc-per-node=${NPROC_PER_NODE} demo.py
# warmup donc on remet a zero les resultats
echo "rank;size;total_loading_time;total_computing_time;total_time" > results/results_cpu.csv

#1
torchrun --nnodes=1 --nproc-per-node=${NPROC_PER_NODE} demo.py
# append dans le fichier results_cpu.csv des 0 pour separer les resultats 
echo ";;;;" >> results/results_cpu.csv

#2
torchrun --nnodes=1 --nproc-per-node=${NPROC_PER_NODE} demo.py
echo ";;;;" >> results/results_cpu.csv

#3
torchrun --nnodes=1 --nproc-per-node=${NPROC_PER_NODE} demo.py
echo ";;;;" >> results/results_cpu.csv

#4
torchrun --nnodes=1 --nproc-per-node=${NPROC_PER_NODE} demo.py
echo ";;;;" >> results/results_cpu.csv  

#5
torchrun --nnodes=1 --nproc-per-node=${NPROC_PER_NODE} demo.py
echo ";;;;" >> results/results_cpu.csv

#6
torchrun --nnodes=1 --nproc-per-node=${NPROC_PER_NODE} demo.py
echo ";;;;" >> results/results_cpu.csv

#7
torchrun --nnodes=1 --nproc-per-node=${NPROC_PER_NODE} demo.py
echo ";;;;" >> results/results_cpu.csv

#8
torchrun --nnodes=1 --nproc-per-node=${NPROC_PER_NODE} demo.py
echo ";;;;" >> results/results_cpu.csv

#9
torchrun --nnodes=1 --nproc-per-node=${NPROC_PER_NODE} demo.py
echo ";;;;" >> results/results_cpu.csv

#10
torchrun --nnodes=1 --nproc-per-node=${NPROC_PER_NODE} demo.py
echo ";;;;" >> results/results_cpu.csv



# echo "<----------------------------------->"
# torchrun --nnodes=1 --nproc-per-node=8 demo_gpu.pygit config pull.rebase true