from dis import show_code
from tkinter.tix import Tree
from typing import Type, List, Tuple
from sklearn.utils import shuffle

import torch
import numpy as np
from sklearn import datasets

from visualize import show_loss_curve, show_decision_boundary


class CircelsDataset:
    def __init__(self, n_samples: int = 10000) -> None:
        points, labels = datasets.make_circles(
            n_samples, shuffle=True, factor=0.5, noise=0.05
        )
        # Hier war ein komischer Faktor davor
        self.points = points.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> Tuple[np.float32, np.float32]:
        return self.points[idx], self.labels[idx]


class BinaryClassifierMLP(torch.nn.Module):
    def __init__(self, n_inputs: int, hidden: List, activation_fn: Type) -> None:
        super().__init__()

        self.layers = torch.nn.ModuleList()

        # Hier waren die Input und Output sizes falsch
        current_inputs = n_inputs

        for i in range(len(hidden)):
            current_outputs = hidden[i]

            self.layers.append(torch.nn.Linear(current_inputs, current_outputs))
            self.layers.append(activation_fn())
            current_inputs = current_outputs

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(current_inputs, 1), torch.nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hier stand y = layer(x), dann würde das x aber nicht weitergegeben werden
        for layer in self.layers:
            x = layer(x)

        return self.classifier(x).squeeze()


def train_binary(
    epochs: int,
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
) -> List:
    loss_curve = []
    loss_fn = torch.nn.BCELoss()

    for _ in range(epochs):
        for _, (input_data, gt_label) in enumerate(dataloader):
            # zero_grad() stand hier direkt vor step(), das geht natürlich nicht
            model.zero_grad()

            prediction = model(input_data)
            loss_val = loss_fn(prediction, gt_label)
            loss_val.backward()

            optim.step()
            loss_curve.append(loss_val.item())

    return loss_curve


@torch.no_grad()
def evaluate(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module,) -> float:
    accuracy = 0.0

    for _, (input_data, gt_label) in enumerate(dataloader):
        prediction = model(input_data)
        accuracy += torch.sum(prediction.round() == gt_label)

    return 100.0 * accuracy / len(dataloader.dataset)


if __name__ == "__main__":
    """
    Aufgabenteil 1
    """
    dataset = CircelsDataset()
    # hier war shuffle=False, das sollte man so aber ja nicht machen
    dataloader = torch.utils.data.DataLoader(dataset, 32, shuffle=True)
    mlp1 = BinaryClassifierMLP(2, [4, 4], torch.nn.Tanh)
    optim = torch.optim.Adam(mlp1.parameters(), lr=0.01)

    # print("Accuarcy before training", evaluate(dataloader, mlp1))
    # loss_curve = train_binary(10, dataloader, mlp1, optim)
    # print("Accuarcy after training", evaluate(dataloader, mlp1))
    # show_loss_curve(loss_curve)
    # show_decision_boundary(mlp1, dataset.points, dataset.labels)

    """
    Aufgabenteil 2
    """
    mlp2 = BinaryClassifierMLP(2, [4] * 2, torch.nn.ReLU)
    optim = torch.optim.Adam(mlp2.parameters(), lr=0.01)

    print("Accuarcy before training", evaluate(dataloader, mlp2))
    loss_curve = train_binary(10, dataloader, mlp2, optim)
    print("Accuarcy after training", evaluate(dataloader, mlp2))
    show_loss_curve(loss_curve)
    show_decision_boundary(mlp2, dataset.points, dataset.labels)

    for layer in mlp2.layers:
        if isinstance(layer, torch.nn.Linear):
            print(layer.weight)

