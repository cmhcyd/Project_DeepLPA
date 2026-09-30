import torch
import torch.nn as nn
import torch.nn.functional as F

class CE(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        return ce_loss.mean()