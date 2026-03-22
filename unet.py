import torch
import torch.nn as nn

def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.ReLU()
    )

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=5):
        super().__init__()

        self.d1 = conv_block(in_channels, 64)
        self.p1 = nn.MaxPool2d(2)

        self.d2 = conv_block(64, 128)
        self.p2 = nn.MaxPool2d(2)

        self.bridge = conv_block(128, 256)

        self.u1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c1 = conv_block(256, 128)

        self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c2 = conv_block(128, 64)

        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))

        b = self.bridge(self.p2(d2))

        u1 = self.u1(b)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.c1(u1)

        u2 = self.u2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.c2(u2)

        return self.out(u2)