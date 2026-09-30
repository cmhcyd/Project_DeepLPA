import torch
import torch.nn as nn
class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, fp_beta=0.005, fn_beta=0.005, signal_class=0, gammap=6, gamman=6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.gamman = gamman
        self.gammap = gammap
        self.fp_beta = fp_beta
        self.fn_beta = fn_beta
        self.signal_class = signal_class
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss).clamp(min=1e-6, max=1-1e-6)
        alpha_factor = torch.where(targets == self.signal_class, self.alpha, 1 - self.alpha)
        focal_term = ce_loss * alpha_factor * (1 - pt)**self.gamma
        # focal_term = ce_loss * (1 - pt)**self.gamma
        probs = torch.softmax(inputs, dim=1)
        signal_probs = probs[:, self.signal_class]
        # fp_penalty = self.fp_beta * (targets != self.signal_class).float() * signal_probs**2
        # fn_penalty = self.fn_beta * (targets == self.signal_class).float() * (1 - signal_probs)**2
        fp_penalty = self.fp_beta * (targets != self.signal_class).float() * (signal_probs)**self.gammap
        fn_penalty = self.fn_beta * (targets == self.signal_class).float() * (1 - signal_probs)**self.gamman
        total_loss = focal_term + fp_penalty + fn_penalty
        return total_loss.mean()
    