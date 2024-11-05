import torch.distributed as dist
import os
os.environ["OMP_NUM_THREADS"] = "1"
dist.init_process_group(backend='gloo', init_method="env://")
print('Hello from process {} (out of {})!'.format(dist.get_rank(), dist.get_world_size()))
