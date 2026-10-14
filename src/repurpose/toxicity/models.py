import torch
from torch import nn


class MultiTaskMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 12),
        )

    def forward(self, x):
        return self.network(x)


def masked_bce(logits, targets, pos_weight=None):
    mask = torch.isfinite(targets)
    safe = torch.nan_to_num(targets, nan=0.0)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, safe, pos_weight=pos_weight, reduction="none"
    )
    return (loss * mask).sum() / mask.sum().clamp_min(1)
