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


def run(rank, size):
    
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
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
    sample_size = 32//size
    dataloader = DataLoader(local_dataset, batch_size=sample_size, shuffle=True)
    model = models.resnet18().to(device_id)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes)).to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)


    print(f"Start running basic DDP example on rank {rank} with model Resnet18.")
    st = time.time()
    train_images, train_labels = next(iter(dataloader))
    train_images = train_images.to(device_id)
    train_labels = train_labels.to(device_id)
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
    dist.init_process_group("nccl", init_method="env://")
    size = dist.get_world_size()
    rank = dist.get_rank()

    loading_time, computing_time = run(rank, size)
    # on stocke le resultat dans un fichier csv pour pouvoir le traiter plus tard
    with open("results/results_gpu.csv", "a") as f:
        f.write(f"{rank};{size};{loading_time};{computing_time};{loading_time+computing_time}\n")