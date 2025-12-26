import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import time
import os
import tarfile
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP


def run(rank, size):
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
    # model = models.resnet18()
    # model = models.vgg19_bn(pretrained=False)
    model = models.vgg19_bn()
    # model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(dataset.classes))
    ddp_model = DDP(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)


    print(f"Start running basic DDP example on rank {rank} with model VGG19_BN.")
    st = time.time()
    train_images, train_labels = next(iter(dataloader))
    et_read = time.time()
    print(f'Loading time: {et_read-st} seconds')
    optimizer.zero_grad()
    outputs = ddp_model(train_images)
    loss_fn(outputs, train_labels).backward()
    et = time.time()
    print(f'Computing + Communication time: {et-et_read} seconds')
    optimizer.step()
    dist.destroy_process_group()
    print(f"Finished running basic DDP example on rank {rank}.")

if __name__ == "__main__":
    dist.init_process_group("gloo", init_method="env://")
    size = dist.get_world_size()
    rank = dist.get_rank()
    run(rank, size)