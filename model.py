import torch
from torch import nn

class DeclippingModel(nn.Module):
    def __init__(self):
        super(DeclippingModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.bottleneck = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),
            # nn.ReLU()
        )
    def forward(self,x):
        x = torch.unsqueeze(x,2)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = torch.squeeze(x,2)
        return x