from typing import Union, List
import torch as th
import numpy as np
import matplotlib.pyplot as plt


@th.no_grad()
def show_synthetic_data(
    points: Union[th.Tensor, np.ndarray], labels: Union[th.Tensor, np.ndarray]
) -> None:
    assert points.ndim == 2 and points.shape[-1] == 2

    # Group points according to their labels
    indices_zeros = np.nonzero(labels == 0)
    indices_ones = np.nonzero(labels == 1)

    # Plot both groups with different colors
    _, ax = plt.subplots(1, 1)
    ax.scatter(points[indices_zeros, 0], points[indices_zeros, 1], c="blue")
    ax.scatter(points[indices_ones, 0], points[indices_ones, 1], c="orange")
    plt.show()


@th.no_grad()
def show_loss_curve(loss_curve: List) -> None:
    plt.plot(loss_curve, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Binäre Kreuzentropy")
    plt.legend()
    plt.show()


@th.no_grad()
def show_decision_boundary(
    model: th.nn.Module,
    points: Union[th.Tensor, np.ndarray],
    labels: Union[th.Tensor, np.ndarray],
) -> None:
    assert points.ndim == 2 and points.shape[-1] == 2

    xx, yy = th.meshgrid(
        [th.linspace(-2.0, 2.0, steps=100), th.linspace(-2.0, 2.0, steps=100)],
        indexing="xy",
    )
    xy = th.stack([xx, yy], dim=-1).reshape(-1, 2)

    prediction = model(xy)
    labels = th.round(prediction)

    _, ax = plt.subplots(1, 1)
    ax.contourf(
        xy[:, 0].reshape(100, 100),
        xy[:, 1].reshape(100, 100),
        labels.reshape(100, 100),
        cmap=plt.cm.jet,
    )

    # Group points according to their labels
    indices_zeros = np.nonzero(labels == 0)
    indices_ones = np.nonzero(labels == 1)

    # Plot both groups with different colors
    ax.scatter(points[indices_zeros, 0], points[indices_zeros, 1], c="blue")
    ax.scatter(points[indices_ones, 0], points[indices_ones, 1], c="orange")

    plt.show()
