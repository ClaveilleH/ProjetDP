## Start
- `torchrun --nnodes=1 --nproc-per-node=2 launch.py`

## Tasks
1. Check the code of launch.py, what are the functions to get the rank of the process and the number of all processes?
2. Check the code of send_message.py. Try to launch the code using "torchrun". The example code achieves what kind of communication?
3. Run the code of broadcast.py  and reduce.py.