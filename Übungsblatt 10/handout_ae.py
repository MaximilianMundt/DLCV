from audioop import add
from re import A
from sklearn.semi_supervised import LabelSpreading
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torchsummary
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime
from timeit import default_timer as timer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = (
    "T-shirt/Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)


def add_noise(inputs):
    noise = torch.randn_like(inputs) * 0.3
    return inputs + noise


class SmallMNIST:
    def __init__(self, root, train, transform, download):
        data = datasets.FashionMNIST(root, train=train, download=download)
        self.transform = transform

        # Sortiere die Bilder und Label anhand der Label
        sorter = torch.argsort(data.targets)
        self.images = data.data[sorter]
        self.labels = data.targets[sorter]

        # Behalte von jeder Klasse nur die ersten 1000 Bilder und Label
        selection = [i * 6000 + j for i in range(10) for j in range(1000)]
        self.images = self.images[selection]
        self.labels = self.labels[selection]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx]


class DownBlock(nn.Module):
    def __init__(self, n_features, activation_fn):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(n_features, 2 * n_features, 3, stride=2, padding=1),
            activation_fn(),
            nn.Conv2d(2 * n_features, 2 * n_features, 3, padding="same"),
            activation_fn(),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 5, padding="same"),
            activation_fn(),
            DownBlock(8, activation_fn),
            DownBlock(16, activation_fn),
            DownBlock(32, activation_fn),
            nn.Conv2d(64, 64, 1, padding="same"),
        )

    def forward(self, x):
        return self.model(x)


class TransposeBlock(nn.Module):
    def __init__(self, n_features, activation_fn):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(n_features, n_features // 2, kernel_size=3), activation_fn(),
        )

    def forward(self, x):
        return self.block(x)


class DecoderTranspose(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()

        self.model = nn.Sequential(
            TransposeBlock(64, activation_fn),
            TransposeBlock(32, activation_fn),
            TransposeBlock(16, activation_fn),
            nn.Conv2d(8, 1, kernel_size=5),
        )

    def forward(self, x):
        return self.model(x)


class UpsampleBlock(nn.Module):
    def __init__(self, n_features, activation_fn):
        super().__init__()

        self.block = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(n_features, n_features // 2, kernel_size=3, padding="same"),
            activation_fn(),
        )

    def forward(self, x):
        return self.block(x)


class DecoderUpsample(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()

        self.model = nn.Sequential(
            UpsampleBlock(64, activation_fn),
            UpsampleBlock(32, activation_fn),
            UpsampleBlock(16, activation_fn),
            nn.Conv2d(8, 1, kernel_size=5),
        )

    def forward(self, x):
        return self.model(x)


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


def train_autoencoder(n_epochs, model, optimizer, trainloader, save=False):
    model.train()
    loss_fn = nn.MSELoss()
    batch_history = []
    epoch_history = []

    for epoch in range(1, n_epochs + 1):
        print(f"Beginning epoch {epoch}/{n_epochs}:", end="")
        start = timer()

        for data in trainloader:
            optimizer.zero_grad()

            images, _ = to_device(data, DEVICE)
            noisy_images = add_noise(images)

            outputs = model(noisy_images)
            loss = loss_fn(outputs, images)
            loss.backward()
            batch_history.append(loss.item())

            optimizer.step()

        size = trainloader.batch_size
        mean_loss = sum(batch_history[-size:]) / size
        epoch_history.append(mean_loss)

        end = timer()
        elapsed_time = end - start

        print(f" loss = {round(mean_loss, 4)}, elapsed time = {round(elapsed_time, 2)} s")

    if save:
        date = datetime.now().strftime("%y%m%d_%H%M")
        torch.save(model, f"autoencoder_{date}.pt")

    return epoch_history


@torch.no_grad()
def evaluate_autoencoder(model, testloader):
    model.eval()

    images, labels = next(iter(testloader))
    noisy_images = add_noise(images)
    reconstructed_images = model.cpu()(noisy_images)
    visualize_images(noisy_images, reconstructed_images, images, labels)


@torch.no_grad()
def visualize_images(noisy_inputs, reconstructed_images, original_images, labels):
    transform_image = lambda image: image.permute(1, 2, 0).numpy()
    _, axs = plt.subplots(len(noisy_inputs), 3, figsize=(len(noisy_inputs) * 2, 8))

    for i, (noisy_input, reconstructed_image, original_image, label) in enumerate(
        zip(noisy_inputs, reconstructed_images, original_images, labels)
    ):
        axs[i][0].imshow(transform_image(noisy_input))
        axs[i][0].set_title("Noisy Input")
        axs[i][1].imshow(transform_image(reconstructed_image))
        axs[i][1].set_title("Reconstructed")
        axs[i][2].imshow(transform_image(original_image))
        axs[i][2].set_title(f"Original ({CLASSES[label]})")

    for axx in axs:
        for axy in axx:
            axy.set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    traindata = datasets.FashionMNIST(
        "./data", train=True, transform=transforms.ToTensor(), download=True
    )
    trainloader = DataLoader(traindata, batch_size=64, shuffle=True, num_workers=6)

    """
    Autoencoder mit Upsampling Decoder 
    """
    # model = AutoEncoder(Encoder(nn.LeakyReLU), DecoderUpsample(nn.ReLU)).to(DEVICE)
    model = torch.load("autoencoder_220624_2108.pt")

    # optimizer = torch.optim.Adam(model.parameters())

    # loss_curve = train_autoencoder(30, model, optimizer, trainloader, save=True)
    # plt.plot(loss_curve)
    # plt.show()

    testdata = datasets.FashionMNIST("./data", train=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(traindata, batch_size=5, shuffle=True)

    evaluate_autoencoder(model, testloader)