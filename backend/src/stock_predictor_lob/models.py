"""Independent PyTorch implementations of the DeepLOB and TransLOB architectures."""

import torch
from torch import nn
from torch.nn import functional as F


class DeepLOB(nn.Module):
    def __init__(self, window: int = 100):
        super().__init__()
        if window < 19:
            raise ValueError("DeepLOB requires at least 19 input observations")
        layers = []
        incoming = 1
        for width, stride in [(2, 2), (2, 2), (10, 1)]:
            for kernel, step in [((1, width), (1, stride)), ((4, 1), 1), ((4, 1), 1)]:
                layers.extend(
                    [
                        nn.Conv2d(incoming, 32, kernel, stride=step),
                        nn.LeakyReLU(0.01),
                    ]
                )
                incoming = 32
            layers.append(nn.BatchNorm2d(32))
        self.features = nn.Sequential(*layers)
        self.branch3 = nn.Sequential(
            nn.Conv2d(32, 64, 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, (3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(32, 64, 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, (5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(32, 64, 1),
            nn.LeakyReLU(0.01),
        )
        self.lstm = nn.LSTM(192, 64, batch_first=True)
        self.output = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x.unsqueeze(1))
        x = torch.cat([self.branch3(x), self.branch5(x), self.branch_pool(x)], dim=1)
        x = x.squeeze(-1).transpose(1, 2)
        x, _ = self.lstm(x)
        return self.output(x[:, -1])


class CausalConv(nn.Module):
    def __init__(self, incoming: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(incoming, 14, 2, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(F.pad(x, (self.dilation, 0))))


class SharedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(15, 3, batch_first=True, dropout=0)
        self.norm1 = nn.LayerNorm(15)
        self.norm2 = nn.LayerNorm(15)
        self.feedforward = nn.Sequential(nn.Linear(15, 60), nn.ReLU(), nn.Linear(60, 15))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + attended)
        return self.norm2(x + self.feedforward(x))


class TransLOB(nn.Module):
    def __init__(self, window: int = 100):
        super().__init__()
        self.convs = nn.Sequential(
            *[
                CausalConv(40 if i == 0 else 14, dilation)
                for i, dilation in enumerate([1, 2, 4, 8, 16])
            ]
        )
        self.norm = nn.LayerNorm(14)
        self.block = SharedAttention()  # The paper shares parameters across both blocks.
        self.register_buffer("position", torch.linspace(-1, 1, window)[None, :, None])
        self.register_buffer("mask", torch.ones(window, window, dtype=torch.bool).triu(1))
        self.hidden = nn.Linear(window * 15, 64)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(64, 3)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.convs(x.transpose(1, 2)).transpose(1, 2))
        x = torch.cat([x, self.position.expand(len(x), -1, -1)], dim=-1)
        for _ in range(2):
            x = self.block(x, self.mask)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x).flatten(1)
        return self.output(self.dropout(F.relu(self.hidden(x))))


def build_model(name: str, window: int = 100) -> nn.Module:
    if name == "deeplob":
        return DeepLOB(window)
    if name == "translob":
        return TransLOB(window)
    raise ValueError("Unknown LOB architecture")
