"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Giao diện:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", smoothing: float = 0.1, gamma: float = 2.0,
                    alpha=None, weight=None):
    """kind: "ce", "ls" (label smoothing), "focal", "ce_weighted" (cần `weight`)."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(smoothing)
    if kind == "focal":
        return FocalLoss(gamma, alpha)
    if kind == "ce_weighted":
        if weight is None:
            raise ValueError("ce_weighted cần weight (xem class_weights)")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(weight, dtype=torch.float32))
    raise ValueError(f"loss không hỗ trợ: {kind}")


class LabelSmoothingCE(nn.Module):
    """CE với nhãn mềm q'(k) = (1 - eps) * 1[k == y] + eps / K. Tự cài đặt (không dùng tham số
    label_smoothing của PyTorch) để kiểm tra lại được bằng tay; eps = 0 cho đúng CE."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.eps = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=1)
        nll = -logp.gather(1, target[:, None]).squeeze(1)
        uniform = -logp.mean(dim=1)  # = sum_k (1/K) * (-log p_k)
        return ((1 - self.eps) * nll + self.eps * uniform).mean()


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t), trung bình theo batch. gamma = 0 -> CE."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=1)
        logpt = logp.gather(1, target[:, None]).squeeze(1)
        pt = logpt.exp()
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.alpha is not None:
            loss = loss * self.alpha.to(loss.device)[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số lớp từ số ảnh mỗi lớp của tập TRAIN.

    beta = 0: w_c = 1 / n_c, chuẩn hoá về trung bình 1.
    beta > 0: class-balanced w_c = (1 - beta) / (1 - beta ** n_c), chuẩn hoá tổng = số lớp.
    """
    n = np.asarray(counts, dtype=np.float64)
    if (n <= 0).any():
        raise ValueError(f"có lớp không có ảnh: {n}")
    if beta == 0:
        w = 1.0 / n
        w = w / w.mean()
    else:
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
        w = w / w.sum() * len(n)
    return torch.tensor(w, dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn batch. lam ~ Beta(alpha, alpha), dùng RNG của numpy (đã cố định trong set_seed).

    cutmix: hộp có cạnh tỉ lệ sqrt(1 - lam), tâm ngẫu nhiên, bị cắt ở biên ảnh; lam được tính lại
    theo diện tích thật của hộp sau khi cắt.
    Trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm].
    """
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x_mix = lam * x + (1 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2:]
        cut = np.sqrt(1.0 - lam)
        ch, cw = int(h * cut), int(w * cut)
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
        x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
    else:
        raise ValueError(f"mode không hỗ trợ: {mode}")
    return x_mix, (y, y[perm], float(lam))


def mixed_loss(criterion, logits, targets):
    """lam * L(logits, y_a) + (1 - lam) * L(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
