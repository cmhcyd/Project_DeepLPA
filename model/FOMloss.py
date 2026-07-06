import torch
import torch.nn as nn

class FOMLoss(nn.Module):
    """
    Loss based on FOM = (S / n_pos) / sqrt(S+B), normalized by number of positive samples.
    S: sum of positive class probabilities
    B: sum of negative class probabilities
    n_pos: number of positive samples
    """
    def __init__(self, signal_class=0, eps=1e-8, normalize_by_pos=True):
        super().__init__()
        self.signal_class = signal_class
        self.eps = eps
        self.normalize_by_pos = normalize_by_pos

    def forward(self, inputs, targets):
        """
        inputs: logits of shape (N, C)
        targets: ground truth labels of shape (N,)
        """
        probs = torch.softmax(inputs, dim=1)
        pos_probs = probs[:, self.signal_class]

        pos_mask = (targets == self.signal_class)
        neg_mask = (targets != self.signal_class)

        S = pos_probs[pos_mask].sum()
        B = pos_probs[neg_mask].sum()
        n_pos = pos_mask.sum().float()

        if n_pos == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        if self.normalize_by_pos:
            S = S / (n_pos + self.eps)

        denom = torch.sqrt(S + B + self.eps)
        fom = S / denom
        loss = -fom
        return loss