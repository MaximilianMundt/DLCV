import os
import math
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from typing import List, Type, Tuple


class FashionMNISTDataset:
    def __init__(self, train: bool, classes: List) -> None:
        self.load_data(train)

        mask = torch.zeros_like(self.targets, dtype=torch.int)
        for label in classes:
            mask += self.targets == label
        indices = torch.nonzero(mask).squeeze()
        self.data = self.data[indices]
        self.targets = self.targets[indices]

        sorter = torch.argsort(self.targets)
        self.data = self.data[sorter]
        self.targets = self.targets[sorter]

    def load_data(self, train: bool) -> None:
        data_folder = "./FashionMNIST"
        if train:
            data_file = "training.pt"
        else:
            data_file = "test.pt"
        self.data, self.targets = torch.load(os.path.join(data_folder, data_file))
        self.data = self.data.float() / 255.0
        self.data = self.data.unsqueeze(1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


class MultiClassifierMLP(torch.nn.Module):
    def __init__(
        self, n_inputs: int, n_outputs: int, hidden: List, activation_fn: Type
    ) -> None:
        super().__init__()

        self.layers = torch.nn.ModuleList()
        pass  # Vervollständigen Sie den Konstruktor der Klasse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x.squeeze()


class ConvBlock(torch.nn.Module):
    def __init__(self, n_inputs: int, activation_fn: Type) -> None:
        super().__init__()

        self.layers = torch.nn.ModuleList()
        pass  # Vervollständigen Sie den Konstruktor der Klasse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MultiClassifierCNN(torch.nn.Module):
    def __init__(
        self,
        img_shape: Tuple,
        n_classes: int,
        n_features: int,
        n_blocks: int,
        activation_fn: Type,
    ) -> None:
        super().__init__()

        self.features = torch.nn.ModuleList()
        self.features.append(
            torch.nn.Conv2d(img_shape[0], n_features, kernel_size=7, padding=3)
        )
        self.features.append(activation_fn())
        # Fügen Sie die weitern Schichten hinzu

        linear_in = 0  # Berechnen Sie die Inputgröße des MLPs
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(linear_in, 128),
            self.layers.append(activation_fn()),
            torch.nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.features:
            x = layer(x)
        return self.classifier(x).squeeze()


def train_multiclass(dataloader, model, optimizer, n_epochs):
    loss_curve = []
    for epoch in range(n_epochs):
        for idx, (imgs, labels) in enumerate(dataloader):
            pass  # Vervollständigen Sie die Funktion
    return loss_curve


def evaluate_multiclass(dataloader, model):
    accuarcy = 0.0
    for imgs, labels in dataloader:
        pass  # Vervollständigen Sie die Funktion
    return accuarcy
