import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss with a designated signal class.
    Args:
        signal_class (int): Positive class index.
        alpha (float): Weight for positive class; negative weight is 1-alpha.
        gamma (float): Focusing parameter, default 2.0.
        reduction (str): 'mean', 'sum', or 'none'.
    """
    def __init__(self, signal_class=0, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.signal_class = signal_class
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: logits of shape (N, C)
            targets: class labels of shape (N,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss).clamp(min=1e-6, max=1-1e-6)

        alpha_factor = torch.where(targets == self.signal_class,
                                    self.alpha,
                                    1 - self.alpha)

        focal_term = alpha_factor * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_term.mean()
        elif self.reduction == 'sum':
            return focal_term.sum()
        else:
            return focal_term