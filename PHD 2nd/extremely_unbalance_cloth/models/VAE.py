import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule
from utils import Conv2d_same,ConvTranspose2d_same  # Conv2d_same与ConvTranspose2d_same默认padding为'same'，Pytorch中nn.Conv2d默认padding为'valid'


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
        x_mean, x_logvar = self.decoder(z)
        return x_mean, x_logvar, z_mean, z_logvar



class Generative(BasicModule):

    def __init__(self, z_dim):
        super(Generative, self).__init__()

        self.model_name = 'generative'

        self.fc = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024,128*5*5),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            ConvTranspose2d_same(128,128,(4,3),(4,3)),nn.ReLU(),
            ConvTranspose2d_same(128,64,3,2),nn.ReLU(),
            ConvTranspose2d_same(64, 32, 3, 2), nn.ReLU(),
            ConvTranspose2d_same(32, 16, 3, 2), nn.ReLU(),
        )
        self.mean_layer = nn.Sequential(
            ConvTranspose2d_same(16, 8, 5, 2), nn.ReLU(),
            ConvTranspose2d_same(8, 1, 7, 2)
        )
        self.logvar_layer = nn.Sequential(
            ConvTranspose2d_same(16, 8, 5, 2), nn.ReLU(),
            ConvTranspose2d_same(8, 1, 7, 2)
        )

    def forward(self, z):
        h = self.fc(z)
        h = self.deconv(h.view(-1, 128, 5, 5))
        mean = torch.sigmoid(self.mean_layer(h))
        logvar = self.logvar_layer(h)
        return mean, logvar



class Inference(BasicModule):

    def __init__(self, z_dim):
        super(Inference, self).__init__()

        self.model_name = 'inference'

        self.feature = nn.Sequential(
            Conv2d_same(1, 8, 3, 1), nn.ReLU(),
            Conv2d_same(8, 8, 3, 1), nn.ReLU(),
            Conv2d_same(8, 8, 3, 2), nn.ReLU(),
            Conv2d_same(8, 16, 3, 1), nn.ReLU(),
            Conv2d_same(16, 16, 3, 1), nn.ReLU(),
            Conv2d_same(16, 16, 3, 2), nn.ReLU(),
            Conv2d_same(16, 32, 3, 1), nn.ReLU(),
            Conv2d_same(32, 32, 3, 1), nn.ReLU(),
            Conv2d_same(32, 32, 3, 2), nn.ReLU(),
            Conv2d_same(32, 64, 3, 1), nn.ReLU(),
            Conv2d_same(64, 64, 3, 1), nn.ReLU(),
            Conv2d_same(64, 64, 3, 2), nn.ReLU(),
            Conv2d_same(64, 128, 3, 2),nn.ReLU(),
            Conv2d_same(128,128,(4,3),(4,3))
        )
        self.linear = nn.Linear(128 * 5 * 5, 1024)
        self.mean_layer = nn.Linear(1024, z_dim)
        self.logvariance_layer = nn.Linear(1024, z_dim)

    def forward(self, x):
        h = self.feature(x)
        h = F.relu(self.linear(h.view(-1, 128 * 5 * 5)))
        mean = self.mean_layer(h)
        logvariance = self.logvariance_layer(h)
        return mean, logvariance
