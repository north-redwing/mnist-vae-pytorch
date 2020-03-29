import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, x_dim, z_dim, device):
        super(VAE, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.fc1 = nn.Linear(x_dim, 400)
        self.fc21 = nn.Linear(400, z_dim)
        self.fc22 = nn.Linear(400, z_dim)
        self.fc3 = nn.Linear(z_dim, 400)
        self.fc4 = nn.Linear(400, x_dim)

    def encoder(self, x):
        x = x.view(-1, self.x_dim)
        x = F.relu(self.fc1(x))
        mean = self.fc21(x)
        logvar = self.fc22(x)
        return mean, logvar

    def sample_z(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(mean.shape, device=self.device)
        z = mean + eps * std
        return z

    def decoder(self, z):
        y = F.relu(self.fc3(z))
        y = self.fc4(y)
        y = torch.sigmoid(y)
        return y

    def forward(self, x):
        x = x.view(-1, self.x_dim)
        mean, logvar = self.encoder(x)
        z = self.sample_z(mean, logvar)
        y = self.decoder(z)
        return y, z

    def loss_function(self, x):
        mean, logvar = self.encoder(x)
        z = self.sample_z(mean, logvar)
        y = self.decoder(z)
        reconstruction = F.binary_cross_entropy(y, x, reduction='sum')
        KL = -0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar))
        return reconstruction + KL
