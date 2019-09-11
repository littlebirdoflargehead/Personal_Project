import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .BasicModule import BasicModule



class VAE(BasicModule):

    def __init__(self,z_dim=20):
        super(VAE,self).__init__()

        self.model_name = 'vae'

        self.z_dim = z_dim

        self.generative = Generative(z_dim)
        self.inference = Inference(z_dim)

    def encoder(self,x):
        mu, logvar = self.inference(x)
        return mu,logvar

    def decoder(self,z):
        x = self.generative(z)
        return torch.sigmoid(x)

    def reparameter_trick(self,mu,logvar):
        epslon = torch.randn_like(logvar)
        std = torch.exp(0.5*logvar)
        z = mu+epslon*std
        return z,epslon

    def forward(self, x):
        mu,logvar = self.encoder(x)
        z,_ = self.reparameter_trick(mu,logvar)
        out = self.decoder(z)
        return out,mu,logvar


class Generative(BasicModule):

    def __init__(self,z_dim):
        super(Generative,self).__init__()

        self.model_name = 'generative'

        self.fc = nn.Sequential(
            nn.Linear(z_dim,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128*6*8),
            nn.BatchNorm1d(128*6*8),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=256,
                kernel_size=8,
                stride=4,
                padding=2,
                bias=False
            ), # n_new = s*(n_old-1)+k-2*p
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256,128,8,4,2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128,1,9,5,2),
        )

    def forward(self, z):
        h = self.fc(z)
        x = self.deconv(h.view(-1,128,8,6))
        return x


class Inference(BasicModule):

    def __init__(self,z_dim):
        super(Inference,self).__init__()

        self.model_name = 'inference'

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=256,
                kernel_size=9,
                stride=5,
                padding=2,
                bias=False
            ), # n_new = (n_old+2*p-k)/s+1
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256,128,8,4,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.linear = nn.Linear(128*6*8,1024)
        self.mean_layer = nn.Linear(1024,z_dim)
        self.logvariance_layer = nn.Linear(1024,z_dim)

    def forward(self, x):
        h1 = self.conv(x)
        h2 = F.relu(self.linear(h1.view(-1,128*6*8)))
        mean = self.mean_layer(h2)
        logvariance = self.logvariance_layer(h2)
        return mean, logvariance