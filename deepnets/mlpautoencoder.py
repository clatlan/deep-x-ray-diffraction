import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class MLPAutoencoder(nn.Module):
    def __init__(self, input_size):
        super(MLPAutoencoder, self).__init__()

        self.input_size = input_size

        self.encoder = nn.Sequential(
                nn.Linear(self.input_size, 750),
                nn.ReLU(),
                nn.Linear(750, 500),
                nn.ReLU(),
                nn.Linear(500, 250),
                nn.ReLU(),
                nn.Linear(250, 125),
                nn.ReLU(),
                nn.Linear(125, 50))
        self.decoder = nn.Sequential(
                nn.Linear(50, 125),
                nn.ReLU(),
                nn.Linear(125, 250),
                nn.ReLU(),
                nn.Linear(250, 500),
                nn.ReLU(),
                nn.Linear(500, 750),
                nn.ReLU(),
                nn.Linear(750, self.input_size),
                nn.ReLU())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 100, stride=50, padding=50),  # b, 16, 40,
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),  # b, 16, 20
            nn.Conv1d(16, 8, 3, stride=2, padding=1),  # b, 8, 10
            nn.ReLU(),
            # nn.MaxPool1d(2, stride=1)  # b, 8, 9 # Downsampling operation
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, 11, stride=1),  # b, 16, 20
            nn.ReLU(),
            nn.ConvTranspose1d(16, 8, 4, stride=2, padding=1),  # b, 8, 40
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, 100, stride=50, padding=25),  # b, 1, 2000
            nn.Tanh() # upsampling operation
        )

    def forward(self, x):
        z = self.encoder(x) # downsample
        x_cap = self.decoder(z)
        return x_cap



class MLPAutoencoder_BN(nn.Module):
    def __init__(self, input_size):
        super(MLPAutoencoder_BN, self).__init__()

        self.input_size = input_size

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 125),
            nn.BatchNorm1d(125),
            nn.ReLU (0.2),
            nn.Linear(125, 25))
        self.decoder = nn.Sequential(
            nn.Linear(25, 125),
            nn.BatchNorm1d(125),
            nn.ReLU(),
            nn.Linear(125, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU (0.2),
            nn.Linear(1000, self.input_size),
            nn.ReLU())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
