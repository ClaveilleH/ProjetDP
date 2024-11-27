import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import time
import torchvision.models as models
import torchvision
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP


def run(rank, size):

    dataset = torchvision.datasets.Imagenette('/data/coati/user/tdasilva/dataset/imagenet64', download=False)
    dataset_size = len(dataset)
    localdataset_size = dataset_size//size
    local_dataset = torch.utils.data.Subset(dataset, range(rank*localdataset_size, (rank+1)*localdataset_size))
    sample_size = 1024//size
    dataloader = DataLoader(local_dataset, batch_size=sample_size, shuffle=True)
    model = models.resnet18()
    ddp_model = DDP(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)


    print(f"Start running basic DDP example on rank {rank} with model Resnet18.")
    st = time.time()
    train_images, train_labels = next(iter(dataloader))
    optimizer.zero_grad()
    outputs = ddp_model(train_images)
    loss_fn(outputs, train_labels).backward()
    et = time.time()
    print(f'Execution time: {et-st} seconds')

    print(f"rank {rank}: with gradients {model.net1.weight.grad}\n")
    optimizer.step()
    dist.destroy_process_group()
    print(f"Finished running basic DDP example on rank {rank}.")

if __name__ == "__main__":
    dist.init_process_group("gloo", init_method="env://")
    size = dist.get_world_size()
    rank = dist.get_rank()
    run(rank, size)