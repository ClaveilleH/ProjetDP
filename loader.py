import torchvision
import torch
import wget
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torchvision import transforms
from utils import experiment_not_implemented_message
from os.path import isdir


data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,],
                             std=[0.229,])
    ])

def get_loader(experiment_name, batch_size=1, is_trainning = True):
    """
    constructs data loader for an experiment

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be used;
        possible are {"mnist"}

    batch_size: int
        the size of the batch

    is_trainning: bool
        the use of training dataset

    Returns
    -------
        DataLoader

    """
    if experiment_name == "faces":
        if not isdir("data/faces/training"):
            url = "http://www-sop.inria.fr/members/Chuan.Xu/faces.zip"
            wget.download(url)
            with zipfile.ZipFile("faces.zip", "r") as zip_ref:
                zip_ref.extractall("./data")
        if is_trainning:
            pathname = "data/faces/training"
        else:
            pathname = "data/faces/testing"
        dataset = torchvision.datasets.ImageFolder(pathname, transform=data_transform)

    else:
        raise NotImplementedError(
            experiment_not_implemented_message(experiment_name=experiment_name)
        )

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return data_loader