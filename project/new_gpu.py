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

# ============================================
# VARIABLES GLOBALES À CONFIGURER
# ============================================
BATCH_SIZES = [16, 32, 64, 128, 256]  # Liste des batch sizes à tester
NUM_ITERATIONS = 5  # Nombre d'itérations pour moyenner les résultats
OUTPUT_FILE = "results/results_gpu.csv"
# ============================================

def run(rank, size, batch_size):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device_id = rank % torch.cuda.device_count()
    
    # --- 1. Set paths ---
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    download_root = "./"
    dataset_folder = os.path.join(download_root, "imagenette2-160")
    
    # --- 2. Download the dataset (only rank 0) ---
    if rank == 0:
        if not os.path.exists(dataset_folder):
            print("Downloading Imagenette...")
            download_url(dataset_url, download_root)
            
            # Extract
            print("Extracting...")
            with tarfile.open(os.path.join(download_root, "imagenette2-160.tgz")) as tar:
                tar.extractall(path=download_root)
            print("Done!")
    
    # Synchronize all processes
    dist.barrier()
    
    # --- 3. Define transforms ---
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # --- 4. Load dataset with ImageFolder ---
    dataset = ImageFolder(root=os.path.join(dataset_folder, "train"), 
                         transform=transform_train)
    dataset_size = len(dataset)
    local_dataset_size = dataset_size // size
    local_dataset = torch.utils.data.Subset(
        dataset, 
        range(rank * local_dataset_size, (rank + 1) * local_dataset_size)
    )
    
    local_batch_size = batch_size // size
    dataloader = DataLoader(local_dataset, batch_size=local_batch_size, shuffle=True)
    
    # --- 5. Create model ---
    model = models.resnet18().to(device_id)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes)).to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    print(f"Start running DDP on rank {rank}/{size} with batch_size {batch_size} on GPU {device_id}")
    
    # --- 6. Training loop ---
    total_loading_time = 0
    total_computing_time = 0
    
    for iteration in range(NUM_ITERATIONS):
        st = time.time()
        train_images, train_labels = next(iter(dataloader))
        train_images = train_images.to(device_id)
        train_labels = train_labels.to(device_id)
        et_read = time.time()
        loading_time = et_read - st
        
        optimizer.zero_grad()
        outputs = ddp_model(train_images)
        loss_fn(outputs, train_labels).backward()
        optimizer.step()
        
        # Synchronize GPU operations
        torch.cuda.synchronize()
        
        et = time.time()
        computing_time = et - et_read
        
        total_loading_time += loading_time
        total_computing_time += computing_time
        
        if rank == 0:
            print(f"Iteration {iteration+1}/{NUM_ITERATIONS} - "
                  f"Loading: {loading_time:.4f}s, Computing: {computing_time:.4f}s")
    
    print(f"Finished running DDP on rank {rank}.")
    return total_loading_time, total_computing_time

def main():
    dist.init_process_group("nccl", init_method="env://")
    size = dist.get_world_size()
    rank = dist.get_rank()
    
    dist.barrier()
    
    # Tester différents batch sizes
    for batch_size in BATCH_SIZES:
        loading_time, computing_time = run(rank, size, batch_size)
        
        # Synchroniser les résultats entre tous les processus
        dist.barrier()
        
        # Seul le rank 0 écrit dans le fichier
        if rank == 0:
            total_time = loading_time + computing_time
            with open(OUTPUT_FILE, "a") as f:
                f.write(f"{size};{batch_size};{loading_time:.6f};{computing_time:.6f};{total_time:.6f}\n")
            print(f"\n=== Results for size={size}, batch_size={batch_size} ===")
            print(f"Total loading time: {loading_time:.4f}s")
            print(f"Total computing time: {computing_time:.4f}s")
            print(f"Total time: {total_time:.4f}s\n")
        
        dist.barrier()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()