import os
import math
from tabnanny import verbose
from numpy import pad
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from typing import List, Type, Tuple


class FashionMNISTDataset:
    def __init__(self, train: bool, classes: List = list(range(10))) -> None:
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
    def __init__(self, n_inputs: int, n_outputs: int, hidden: List, activation_fn: Type) -> None:
        super().__init__()

        self.layers = torch.nn.ModuleList()
        pass  # Vervollständigen Sie den Konstruktor der Klasse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x.squeeze()


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, filters: int, activation_fn: Type) -> None:
        super().__init__()

        conv1 = torch.nn.Conv2d(in_channels, filters, kernel_size=3, stride=1, padding="same")
        conv2 = torch.nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding="same")

        self.layers = torch.nn.ModuleList()
        self.layers.append(conv1)
        self.layers.append(activation_fn())
        self.layers.append(conv2)
        self.layers.append(activation_fn())
        self.layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x


class MultiClassifierCNN(torch.nn.Module):
    def __init__(self, img_shape: Tuple, n_classes: int, activation_fn: Type,) -> None:
        super().__init__()

        self.features = torch.nn.Sequential(
            ConvBlock(img_shape[-1], 64, activation_fn), ConvBlock(64, 128, activation_fn),
        )

        out_width = img_shape[0] // 4
        out_height = img_shape[1] // 4
        linear_in = 128 * out_width * out_height

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(linear_in, 128), activation_fn(), torch.nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x


def train_multiclass(dataloader, model, optimizer, n_epochs, verbose=True):
    loss_curve = []
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        for imgs, labels in dataloader:
            model.zero_grad()

            prediction = model(imgs)
            loss_val = loss_fn(prediction, labels)
            loss_val.backward()

            optimizer.step()
            loss_curve.append(loss_val.item())

        if verbose:
            print(f"Epoch {epoch+1}/{n_epochs}: CrossEntropy = {loss_curve[-1]:.4f}")

    return loss_curve


def evaluate_multiclass(dataloader, model):
    accuracy = 0.0

    for input_data, gt_label in dataloader:
        prediction = model(input_data)
        accuracy += (torch.argmax(prediction) == gt_label).float().sum()

    return (accuracy / len(dataloader.dataset)).item()
