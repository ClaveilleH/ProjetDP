import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import time
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
from torch.utils.data import DataLoader
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
import tarfile

import sys

def run(rank, size, batch_size=32):
    # determine local rank and available GPUs
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    available_gpus = torch.cuda.device_count()
    # determine local requested parallelism (torchrun sets LOCAL_WORLD_SIZE or use NPROC_PER_NODE)
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', os.environ.get('NPROC_PER_NODE', '1')))

    # fail fast if not enough GPUs for requested local world size
    if available_gpus < local_world_size and local_world_size > 1:
        raise RuntimeError(f"Not enough GPUs for requested local world size: requested {local_world_size}, available {available_gpus}")

    if available_gpus > 0:
        # choose a GPU for this local rank
        device = torch.device(f"cuda:{local_rank % available_gpus}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    # --- 1. Set paths ---
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    download_root = "./"
    dataset_folder = os.path.join(download_root, "imagenette2-160")

    # --- 2. Download the dataset ---
    if not os.path.exists(dataset_folder):
        print("Downloading Imagenette...")
        download_url(dataset_url, download_root)
        # Extract
        print("Extracting...")
        with tarfile.open(os.path.join(download_root, "imagenette2-160.tgz")) as tar:
            tar.extractall(path=download_root)
        print("Done!")

    # --- 3. Define transforms ---
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # --- 4. Load dataset with ImageFolder ---
    dataset = ImageFolder(root=os.path.join(dataset_folder, "train"), transform=transform_train)

    dataset_size = len(dataset)
    localdataset_size = dataset_size//size
    local_dataset = torch.utils.data.Subset(dataset, range(rank*localdataset_size, (rank+1)*localdataset_size))
    sample_size = max(1, batch_size // size)
    dataloader = DataLoader(local_dataset, batch_size=sample_size, shuffle=True)

    # create model and move to the selected device
    model = models.vgg19_bn().to(device)
    # VGG has classifier (nn.Sequential), replace last linear layer
    if hasattr(model, 'classifier'):
        try:
            in_feats = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_feats, len(dataset.classes)).to(device)
        except Exception:
            # fallback if structure differs
            pass

    # Wrap model in DDP. If using CUDA, provide device_ids; otherwise CPU mode works without device_ids.
    if device.type == 'cuda':
        ddp_model = DDP(model, device_ids=[local_rank % available_gpus])
    else:
        ddp_model = DDP(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)


    print(f"Start running basic DDP example on rank {rank} with model Resnet18.")
    st = time.time()
    train_images, train_labels = next(iter(dataloader))
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    et_read = time.time()
    print(f'Loading time: {et_read-st} seconds')
    loading_time = et_read - st
    optimizer.zero_grad()
    outputs = ddp_model(train_images)
    loss_fn(outputs, train_labels).backward()
    et = time.time()
    print(f'Computing + Communication time: {et-et_read} seconds')
    computing_time = et - et_read
    optimizer.step()
    dist.destroy_process_group()
    print(f"Finished running basic DDP example on rank {rank}.")
    return loading_time, computing_time
    
if __name__ == "__main__":

    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 32

    # determine local requested parallelism for validation
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', os.environ.get('NPROC_PER_NODE', '1')))
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if available_gpus < local_world_size and local_world_size > 1:
        raise RuntimeError(f"Not enough GPUs for requested local world size: requested {local_world_size}, available {available_gpus}")

    backend = 'nccl' if available_gpus >= local_world_size and available_gpus > 0 else 'gloo'
    dist.init_process_group(backend, init_method="env://")
    size = dist.get_world_size()
    rank = dist.get_rank()

    loading_time, computing_time = run(rank, size, batch_size)
    # on stocke le resultat dans un fichier csv pour pouvoir le traiter plus tard
    with open("results/results_gpu.csv", "a") as f:
        f.write(f"{rank};{size};{batch_size};{loading_time};{computing_time};{loading_time+computing_time}\n")