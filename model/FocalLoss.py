import torch
import torch.nn as nn
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, signal_class=0, clamp=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.clamp = clamp
        self.signal_class = signal_class
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss).clamp(min=self.clamp, max=1-self.clamp)
        alpha_factor = torch.where(targets == self.signal_class, self.alpha, 1 - self.alpha)
        focal_term = ce_loss * alpha_factor * (1 - pt)**self.gamma
        total_loss = focal_term
        return total_loss.mean()
    