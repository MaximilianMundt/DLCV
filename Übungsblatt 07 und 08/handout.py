import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms

def create_augmented_dataset(train):
    augmentations = transforms.Compose([
        transforms.ToTensor(),
        # Fügen Sie Transformationen hinzu
    ])

    return datasets.CIFAR10(
        "./data", 
        train=train, 
        transform=augmentations,
        download=True
    )