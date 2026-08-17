from __future__ import annotations

import math

import torch
import torch.nn as nn


class WingLoss(nn.Module):
    def __init__(self, w: float = 1.5, epsilon: float = 0.5):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = self.w - self.w * math.log(1.0 + self.w / self.epsilon)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y = torch.abs(prediction - target)
        loss = torch.where(
            y < self.w,
            self.w * torch.log(1.0 + y / self.epsilon),
            y - self.C,
        )
        return loss


class MultifacetedLoss(nn.Module):
    """
    Masked Wing loss for landmark regression:
    - prediction, target: (B, L*2)
    - mask: (B, L) with 1 for valid, 0 for invalid/missing
    """

    def __init__(self, w: float = 0.1, epsilon: float = 0.05):
        super().__init__()
        self.wing_loss = WingLoss(w, epsilon)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, dim = prediction.shape
        L = dim // 2

        loss = self.wing_loss(prediction, target).view(B, L, 2).mean(dim=2)  # (B, L)
        if mask is not None:
            valid_sum = mask.sum().clamp_min(1.0)
            loss = (loss * mask).sum() / valid_sum
        else:
            loss = loss.mean()
        return loss


