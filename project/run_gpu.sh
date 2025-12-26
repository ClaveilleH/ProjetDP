module load conda
conda activate ~/.conda/envs/pytorch/
# torchrun --nnodes=1 --nproc-per-node=1 demo_gpu.py
# echo "<----------------------------------->"
# torchrun --nnodes=1 --nproc-per-node=2 demo_gpu.py

#1
torchrun --nnodes=1 --nproc-per-node=4 demo.py
# append dans le fichier results_cpu.csv des 0 pour separer les resultats 
echo ";;;;" >> results/results_cpu.csv

#2
torchrun --nnodes=1 --nproc-per-node=4 demo.py
echo ";;;;" >> results/results_cpu.csv

#3
torchrun --nnodes=1 --nproc-per-node=4 demo.py
echo ";;;;" >> results/results_cpu.csv

#4
torchrun --nnodes=1 --nproc-per-node=4 demo.py
echo ";;;;" >> results/results_cpu.csv  

#5
torchrun --nnodes=1 --nproc-per-node=4 demo.py
echo ";;;;" >> results/results_cpu.csv

#6
torchrun --nnodes=1 --nproc-per-node=4 demo.py
echo ";;;;" >> results/results_cpu.csv

#7
torchrun --nnodes=1 --nproc-per-node=4 demo.py
echo ";;;;" >> results/results_cpu.csv

#8
torchrun --nnodes=1 --nproc-per-node=4 demo.py
echo ";;;;" >> results/results_cpu.csv

#9
torchrun --nnodes=1 --nproc-per-node=4 demo.py
echo ";;;;" >> results/results_cpu.csv

#10
torchrun --nnodes=1 --nproc-per-node=4 demo.py
echo ";;;;" >> results/results_cpu.csv



# echo "<----------------------------------->"
# torchrun --nnodes=1 --nproc-per-node=8 demo_gpu.pygit config pull.rebase true