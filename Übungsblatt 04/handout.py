from statistics import mode
from tkinter.tix import InputOnly
import torch
import numpy as np
from typing import Type, List, Tuple
from sklearn import datasets


class CircelsDataset:
    def __init__(self, n_samples: int = 10000) -> None:
        self.points, self.labels = datasets.make_circles(
            n_samples, factor=0.5, noise=0.05
        )

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index: int) -> Tuple[np.float32, np.float32]:
        return (self.points[index], self.labels[index])


class BinaryClassifierMLP(torch.nn.Module):
    def __init__(self, n_inputs: int, layers: List, activation_fn: Type) -> None:
        super().__init__()

        self.layers = torch.nn.ModuleList()
        current_inputs = n_inputs

        for i in range(len(layers)):
            self.layers.append(torch.nn.Linear(current_inputs, layers[i]))
            self.layers.append(activation_fn())
            current_inputs = layers[i]

        self.layers.append(torch.nn.Linear(layers[-1], 1))
        self.layers.append(torch.nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x.squeeze()


def train_binary(
    epochs: int,
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
) -> List:
    loss_curve = []

    for epoch in range(epochs):
        for batch_number, (inputs, labels) in enumerate(dataloader):
            optim.zero_grad()

            predictions = model(inputs.float())
            loss_fn = torch.nn.BCELoss()
            loss = loss_fn(predictions, labels.float())
            loss.backward()

            optim.step()
            loss_curve.append(loss.item())

    print("Finished Training")

    return loss_curve


@torch.no_grad()
def evaluate(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
) -> float:
    accuracy = 0.0

    for step, (input_data, gt_label) in enumerate(dataloader):
        pass  # Ersetzen Sie pass durch Ihren code

    return accuracy
