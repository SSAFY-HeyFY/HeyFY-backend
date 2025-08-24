import torch
import torch.nn as nn
import torch.nn.functional as F

class GapWeightedHuber(nn.Module):
    def __init__(self, delta=1.0, gamma=1.0, eps=1e-6):
        """
        delta: Huber 임계값
        gamma: 가중치 지수 (|gap|^gamma)
        eps:   가중치 안정화 상수
        """
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred_scaled_residual, true_scaled_residual, gap_unscaled):
        # pred/true: (B, 1)  (표준화된 residual)
        # gap_unscaled: (B, 1)  (원 단위가 아니라 "수익률" 원시값, 스케일링 안 함)
        error = pred_scaled_residual - true_scaled_residual  # (B,1)

        # Huber in scaled space
        abs_err = torch.abs(error)
        quad = 0.5 * (error ** 2)           # |e| <= delta
        lin  = self.delta * (abs_err - 0.5 * self.delta)  # |e| > delta
        huber = torch.where(abs_err <= self.delta, quad, lin)  # (B,1)

        # gap-weight
        weight = torch.pow(torch.abs(gap_unscaled) + self.eps, self.gamma)  # (B,1)
        loss = (weight * huber).mean()
        return loss
