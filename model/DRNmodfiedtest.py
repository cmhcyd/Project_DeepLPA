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

class FeatureInteractionBlockX(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.SiLU(),
            # nn.GELU(),
            SelfAttentionWrapper(in_dim, num_heads),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(dropout)
        )
        
        self.shortcut = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        ) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class SelfAttentionWrapper(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, x):
        x = x.unsqueeze(0)
        attn_output, _ = self.attention(x, x, x)
        return attn_output.squeeze(0)

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
    def __init__(self, in_dim, num_classes, heads=8, width=256, width2=64, depth=4, dropout=0.1, task="classification"):
        super().__init__()
        self.task = task
        self.stem = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.featureinter = FeatureInteractionBlockX(width, width, heads, dropout)
        blocks = []
        for _ in range(depth):
            blocks.append(ResidualBlock3(width, width, dropout))
        self.blocks = nn.Sequential(*blocks)
        # self.head = nn.Linear(width, num_classes)
        self.head = AdaptiveHead(width, width2, num_classes, dropout)

    def forward(self, x):
        x = self.stem(x)
        x = self.featureinter(x)
        x = self.blocks(x)
        x = self.head(x)
        return x