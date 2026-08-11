"""保持影像尺寸不变的三层卷积分割网络。"""

from torch import nn


class TinySegNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=1),
        )

    def forward(self, x):
        return self.layers(x)
