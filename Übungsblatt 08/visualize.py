from typing import Union, List
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import math


@th.no_grad()
def show_loss_curve(loss_curve: List) -> None:
    plt.plot(loss_curve, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy")
    plt.legend()
    plt.show()


@th.no_grad()
def show_image_grid(images: th.Tensor, labels: th.Tensor, predictions: th.Tensor = None) -> None:
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    size = math.ceil(math.sqrt(len(images)))
    _, axs = plt.subplots(size, size, figsize=(size * 2, size * 2))

    for x in axs:
        for y in x:
            y.set_axis_off()

    for idx, (img, label, pred) in enumerate(zip(images, labels, predictions)):
        pos = idx // size, idx % size
        axs[pos].imshow(img.squeeze().permute(1, 2, 0).numpy())

        if predictions is None:
            axs[pos].set_title(f"Actual: {classes[label]}")
        else:
            axs[pos].set_title(f"Actual: {classes[label]} \n Prediction: {classes[pred]}")

    plt.tight_layout()
    plt.show()
