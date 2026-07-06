import torch
import torch.nn as nn

class AdaptiveHead(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, out_dim),
        )
        
    def forward(self, x):
        return self.head(x)

class ResidualBlock3(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        self.proj = None
        if in_dim != hidden_dim:
            self.proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.drop(out)

        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        out = self.act(out)
        return out

class DRNmod(nn.Module):
    def __init__(self, in_dim, num_classes, width=256, width2=64, depth=4, dropout=0.1, task="classification", **kwargs):
        super().__init__()
        self.task = task
        self.stem = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        blocks = []
        for _ in range(depth):
            blocks.append(ResidualBlock3(width, width, dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head = AdaptiveHead(width, width2, num_classes, dropout)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x