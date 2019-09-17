import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule
from utils import Conv2d_same, \
    ConvTranspose2d_same  # Conv2d_same与ConvTranspose2d_same默认padding为'same'，Pytorch中nn.Conv2d默认padding为'valid'


class VAE(BasicModule):

    def __init__(self, z_dim=20):
        super(VAE, self).__init__()

        self.model_name = 'vae'

        self.z_dim = z_dim

        self.generative = Generative(z_dim)
        self.inference = Inference(z_dim)

    def encoder(self, x):
        z_mean, z_logvar = self.inference(x)
        return z_mean, z_logvar

    def decoder(self, z):
        x_mean, x_logvar = self.generative(z)
        return x_mean, x_logvar

    def reparameter_trick(self, mu, logvar):
        epslon = torch.randn_like(logvar)
        std = torch.exp(0.5 * logvar)
        z = mu + epslon * std
        return z, epslon

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z, _ = self.reparameter_trick(z_mean, z_logvar)
        self.z_dim = z.shape[1:]
        x_mean, x_logvar = self.decoder(z)
        return x_mean, x_logvar, z_mean, z_logvar


class Generative(BasicModule):

    def __init__(self, z_dim):
        super(Generative, self).__init__()

        self.model_name = 'generative'

        self.deconv = nn.Sequential(
            ConvTranspose2d_same(128, 64, 3, 2), nn.ReLU(),
            ConvTranspose2d_same(64, 32, 3, 2), nn.ReLU(),
            ConvTranspose2d_same(32, 16, 3, 2), nn.ReLU(),
            ConvTranspose2d_same(16, 8, 5, 2), nn.ReLU(),
            ConvTranspose2d_same(8, 1, 7, 2)
        )

    def forward(self, z):
        mean = self.deconv(z)
        logvar = torch.tensor(-4.5).to(z.device)
        return mean, logvar


class Inference(BasicModule):

    def __init__(self, z_dim):
        super(Inference, self).__init__()

        self.model_name = 'inference'

        self.feature = nn.Sequential(
            Conv2d_same(1, 8, 3, 1), nn.ReLU(),nn.BatchNorm2d(8),
            Conv2d_same(8, 8, 3, 1), nn.ReLU(),nn.BatchNorm2d(8),
            Conv2d_same(8, 8, 3, 2), nn.ReLU(),nn.BatchNorm2d(8),
            Conv2d_same(8, 16, 3, 1), nn.ReLU(),nn.BatchNorm2d(16),
            Conv2d_same(16, 16, 3, 1), nn.ReLU(),nn.BatchNorm2d(16),
            Conv2d_same(16, 16, 3, 2), nn.ReLU(),nn.BatchNorm2d(16),
            Conv2d_same(16, 32, 3, 1), nn.ReLU(),nn.BatchNorm2d(32),
            Conv2d_same(32, 32, 3, 1), nn.ReLU(),nn.BatchNorm2d(32),
            Conv2d_same(32, 32, 3, 2), nn.ReLU(),nn.BatchNorm2d(32),
            Conv2d_same(32, 64, 3, 1), nn.ReLU(),nn.BatchNorm2d(64),
            Conv2d_same(64, 64, 3, 1), nn.ReLU(),nn.BatchNorm2d(64),
            Conv2d_same(64, 64, 3, 2), nn.ReLU(),nn.BatchNorm2d(64),
        )
        self.mean_layer = Conv2d_same(64,128,3,2)
        self.logvariance_layer = Conv2d_same(64,128,3,2)

    def forward(self, x):
        h = self.feature(x)
        mean = self.mean_layer(h)
        logvariance = self.logvariance_layer(h)
        return mean, logvariance
