from statistics import mode
from tkinter.tix import InputOnly
from xmlrpc.client import Boolean
import torch
import numpy as np
from typing import Type, List, Tuple
from sklearn import datasets


class CircelsDataset:
    def __init__(self, n_samples: int = 10000) -> None:
        points, labels = datasets.make_circles(
            n_samples=n_samples, factor=0.5, noise=0.05
        )
        self.points = points.astype(np.float32)
        self.labels = labels.astype(np.float32)

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
    verbose: bool = True,
) -> List:
    loss_curve = []
    current_loss = 0

    for epoch in range(1, epochs + 1):
        for batch_number, (inputs, labels) in enumerate(dataloader):
            optim.zero_grad()

            predictions = model(inputs)
            loss_fn = torch.nn.BCELoss()
            loss = loss_fn(predictions, labels)
            loss.backward()
            current_loss = loss.item()

            optim.step()

        loss_curve.append(current_loss)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"Epoch {epoch}/{epochs}: loss =", current_loss)

    print("Finished Training")

    return loss_curve


@torch.no_grad()
def evaluate(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
) -> float:
    x = 0

    for step, (inputs, labels) in enumerate(dataloader):
        for input, label in zip(inputs, labels):
            prediction = model(input)
            true_false_prediction = torch.round(prediction)

            x += int(true_false_prediction == label)

    accuracy = 100 / len(dataloader.dataset) * x

    return accuracy
