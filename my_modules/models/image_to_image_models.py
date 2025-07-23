import numpy as np
import torch
from torch import nn


class AutoEncoderMLP(nn.Module):
    def __init__(self, image_size, mask_size, latent_dim):
        super(AutoEncoderMLP, self).__init__()
        self.name = 'AutoEncoder MLP'
        self.image_size = image_size
        self.mask_size = mask_size
        self.latent_dim = latent_dim

        # Encoder
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(np.prod(self.image_size), 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.latent_dim)

        # Decoder
        self.fc4 = nn.Linear(self.latent_dim, 128)
        self.fc5 = nn.Linear(128, 512)
        self.fc6 = nn.Linear(512, np.prod(self.mask_size))

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x[torch.isnan(x)] = 0
        x = self.flat(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.sigm(self.fc6(x))
        x = x.reshape(x.shape[0], *self.mask_size)
        return x


class VariationalAutoEncoderMLP(nn.Module):
    def __init__(self, image_size, latent_dim):
        super(VariationalAutoEncoderMLP, self).__init__()
        self.name = 'Variational AutoEncoder MLP'
        self.image_size = image_size
        self.latent_dim = latent_dim

        # Encoder
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(np.prod(self.image_size), 512)
        self.fc21 = nn.Linear(512, self.latent_dim)  # Mean
        self.fc22 = nn.Linear(512, self.latent_dim)  # Variance

        # Decoder
        self.fc3 = nn.Linear(self.latent_dim, 128)
        self.fc4 = nn.Linear(128, 512)
        self.fc5 = nn.Linear(512, np.prod(self.image_size))

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def encode(self, x):
        x = self.flat(x)
        x = self.relu(self.fc1(x))
        mu = self.fc21(x)
        log_var = self.fc22(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.relu(self.fc3(z))
        x = self.relu(self.fc4(x))
        x = self.sigm(self.fc5(x))
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, output_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2
