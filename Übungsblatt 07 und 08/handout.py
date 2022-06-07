import torch
import torch.nn as nn
import torchsummary
from torchvision import datasets
from torchvision import transforms
from datetime import datetime


class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.Sequential(
            *self._create_conv_block(3, 32),
            *self._create_conv_block(32, 64),
            *self._create_conv_block(64, 128),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def _create_conv_block(self, in_channels, out_channels):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        ]

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = self.fully_connected(x)

        return x

    def summary(self):
        torchsummary.summary(self, (3, 32, 32))


def train(model, dataloader, epochs=10, save=False):
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    cross_entropy = torch.nn.CrossEntropyLoss()
    loss_history = []

    for epoch in range(1, epochs + 1):
        running_loss = []
        print(f"Beginning epoch {epoch} / {epochs}")

        for _, (images, labels) in enumerate(dataloader, 1):
            optimizer.zero_grad()
            outputs = model(images)
            loss = cross_entropy(outputs, labels)
            loss.backward()
            running_loss.append(loss.item())
            optimizer.step()

        epoch_loss = torch.Tensor(running_loss).mean().item()
        loss_history.append(epoch_loss)

    if save:
        date = datetime.strftime("%y%m%d_%H%M")
        torch.save(model, f"model_{date}.pt")

    return loss_history


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    for images, labels in dataloader:
        outputs = model(images)
        predicted_labels = torch.argmax(outputs.data, 1)

        total += images.size(0)
        correct += (predicted_labels == labels).sum().item()

    return round(correct / total, 4)


def create_augmented_dataset(train):
    transformations = [transforms.ToTensor()]

    if train:
        transformations.append(transforms.RandomRotation(degrees=30))
        transformations.append(transforms.RandomHorizontalFlip())

    augmentations = transforms.Compose(transformations)

    return datasets.CIFAR10("./data", train=train, transform=augmentations, download=True)


if __name__ == "__main__":
    model = CIFAR10Classifier()
    model.summary()
