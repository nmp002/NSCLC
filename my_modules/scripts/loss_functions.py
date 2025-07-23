import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        match self.reduction:
            case 'mean':
                return torch.mean(focal_loss)
            case 'sum':
                return torch.sum(focal_loss)
            case 'none':
                return focal_loss
            case _:
                raise ValueError(f'Unknown reduction {self.reduction}')
