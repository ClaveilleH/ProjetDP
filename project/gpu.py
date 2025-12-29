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
    # Determine local rank (set by torchrun) and available GPUs
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0 and local_rank < n_gpus:
        device = torch.device(f"cuda:{local_rank}")
        # set current CUDA device
        torch.cuda.set_device(local_rank)
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
    # VGG uses `classifier`, not `fc` (resnet uses `fc`). Patch classifier head accordingly.
    if hasattr(model, 'classifier') and isinstance(model.classifier, (list, tuple)) or hasattr(model, 'classifier'):
        try:
            # classifier is usually an nn.Sequential; replace the last linear layer
            in_feats = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_feats, len(dataset.classes)).to(device)
        except Exception:
            # fallback: try attribute 'fc'
            if hasattr(model, 'fc'):
                model.fc = nn.Linear(model.fc.in_features, len(dataset.classes)).to(device)
    elif hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, len(dataset.classes)).to(device)

    # Wrap model in DDP. If using CUDA, provide device_ids; otherwise let DDP decide (CPU mode uses no device_ids).
    if device.type == 'cuda':
        ddp_model = DDP(model, device_ids=[local_rank])
    else:
        ddp_model = DDP(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)


    print(f"Start running basic DDP example on rank {rank} with model VGG; device={device}.")
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

    # choose backend: use NCCL only when enough local GPUs exist for the local processes,
    # otherwise fall back to GLOO (works on CPU and mixed setups).
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', os.environ.get('LOCAL_WORLD_SIZE', '1')))
    if torch.cuda.is_available() and torch.cuda.device_count() >= local_world_size:
        backend = 'nccl'
    else:
        backend = 'gloo'
    dist.init_process_group(backend, init_method="env://")
    size = dist.get_world_size()
    rank = dist.get_rank()

    loading_time, computing_time = run(rank, size, batch_size)
    # on stocke le resultat dans un fichier csv pour pouvoir le traiter plus tard
    with open("results/results_gpu.csv", "a") as f:
        f.write(f"{rank};{size};{loading_time};{computing_time};{loading_time+computing_time}\n")