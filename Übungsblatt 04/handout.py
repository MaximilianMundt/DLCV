from typing import Type, List, Tuple
import sklearn

import torch
import numpy as np
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
        pass  # Ersetzen Sie pass durch Ihren code

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
        for batch_idx, (input_data, gt_label) in enumerate(dataloader):
            pass  # Ersetzen Sie pass durch Ihren code
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
