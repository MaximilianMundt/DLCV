from typing import Type, List, Tuple

import torch
import numpy as np
from sklearn import datasets

class CircelsDataset:

    def __init__(self, n_samples : int = 10000) -> None:
        pass # Ersetzen Sie pass durch Ihren code 

    def __len__(self) -> int:
        pass # Ersetzen Sie pass durch Ihren code

    def __getitem__(self, idx : int) -> Tuple[np.float32, np.float32]:
        pass # Ersetzen Sie pass durch Ihren code

class BinaryClassifierMLP(torch.nn.Module):

    def __init__(self, n_inputs : int, layers : List, activation_fn : Type) -> None:
        super().__init__()

        self.layers = torch.nn.ModuleList()
        pass # Ersetzen Sie pass durch Ihren code

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x.squeeze()

def train_binary(
    epochs     : int,
    dataloader : torch.utils.data.DataLoader,
    model      : torch.nn.Module,
    optim      : torch.optim.Optimizer
) -> List:
    loss_curve = []
    for epoch in range(epochs):
        for batch_idx, (input_data, gt_label) in enumerate(dataloader):
            pass # Ersetzen Sie pass durch Ihren code
    return loss_curve

@torch.no_grad()
def evaluate(
    dataloader : torch.utils.data.DataLoader,
    model      : torch.nn.Module,
) -> float:
    accuracy = 0.
    for step, (input_data, gt_label) in enumerate(dataloader):
        pass # Ersetzen Sie pass durch Ihren code
    return accuracy
