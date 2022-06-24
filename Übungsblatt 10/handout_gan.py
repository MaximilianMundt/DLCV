import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchsummary

# Bildet einen Tensor [50, 1, 1] auf ein Bild [1,28,28] ab
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # Block 1
            nn.ConvTranspose2d(50, 64, 4, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Block 2
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Block 3
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Output
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=3, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


# Ein Bild auf ein Bild [1,28,28] auf einen Tensor [1,1,1] ab
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
            # Block 2
            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            # Block 3
            nn.Conv2d(64, 64, 4, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            # Output
            nn.Conv2d(64, 1, 4, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def train_gan(num_epochs, dataloader, netD, optimizerD, netG, optimizerG):
    G_losses, D_losses, img_list = [], [], []

    # Implementieren Sie den GAN trainings algorithmus (Vorlesung 9 Folie 28)

    return G_losses, D_losses, img_list


netD = Discriminator()
netG = Generator()

torchsummary.summary(netD, [1, 28, 28])
torchsummary.summary(netG, [50, 1, 1])

# Setup Adam optimizers for both G and D
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.001, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.001, betas=(0.5, 0.999))

traindata = datasets.FashionMNIST(
    "./data", train=True, transform=transforms.ToTensor(), download=True
)
trainloader = torch.utils.data.DataLoader(traindata, batch_size=128, shuffle=True)
G_losses, D_losses, img_list = train_gan(10, trainloader, netD, optimizerD, netG, optimizerG)

plt.plot(G_losses)
plt.show()

plt.plot(D_losses)
plt.show()

for img in img_list:
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
