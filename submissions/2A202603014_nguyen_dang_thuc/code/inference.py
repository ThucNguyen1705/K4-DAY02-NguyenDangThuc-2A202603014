"""inference.py - các phương pháp suy luận (Bước 3).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test.

Giao diện:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    predict_views(model, loader, device, views)      -> (filenames, y_true, [logits mỗi view])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _run(model, x, amp):
    with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp and x.is_cuda):
        return model(x).float()


def predict_logits(model, loader, device, view=None, amp: bool = False):
    """Chạy model trên loader (thứ tự file giữ nguyên). `view`: hàm biến đổi batch, hoặc None."""
    names, y, outs = predict_views(model, loader, device, (lambda x: [view(x)]) if view else None, amp)
    return names, y, outs[0]


def predict_views(model, loader, device, views=None, amp: bool = False):
    """Một lượt qua loader, mỗi batch tạo K view bằng `views(x) -> list[tensor]`.

    Trả về (filenames, y_true, list K mảng logit). Dùng cho TTA lật, nhiều crop, nhiều tỉ lệ.
    """
    model.eval()
    names, ys, outs = [], [], None
    with torch.inference_mode():
        for x, y, fn in loader:
            x = x.to(device, non_blocking=True)
            batches = views(x) if views else [x]
            res = [_run(model, b, amp).cpu() for b in batches]
            if outs is None:
                outs = [[] for _ in res]
            for i, r in enumerate(res):
                outs[i].append(r)
            ys.append(y)
            names.extend(fn)
    return names, torch.cat(ys).numpy(), [torch.cat(o).numpy() for o in outs]


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W): đảo chiều rộng."""
    return torch.flip(x, dims=[3])


def views_hflip(x):
    return [x, view_hflip(x)]


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) cỡ `crop` từ batch ảnh đầy đủ; flip=True thêm bản lật (10 crop)."""
    h, w = x.shape[-2:]
    tops = [0, 0, h - crop, h - crop, (h - crop) // 2]
    lefts = [0, w - crop, 0, w - crop, (w - crop) // 2]
    crops = [x[..., t:t + crop, l:l + crop] for t, l in zip(tops, lefts)]
    if flip:
        crops += [view_hflip(c) for c in crops]
    return crops


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước. Chỉ dùng cho CNN có global pooling; ViT/Swin cần
    nội suy pos-embed / cửa sổ nên không áp dụng trong bài này."""
    return [F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False, antialias=True)
            if s != x.shape[-1] else x for s in sizes]


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K view: "prob" = trung bình softmax; "logit" = softmax của trung bình logit."""
    if space == "prob":
        return np.mean([_softmax(l) for l in logits_per_view], axis=0)
    if space == "logit":
        return _softmax(np.mean(logits_per_view, axis=0))
    raise ValueError(f"space không hỗ trợ: {space}")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (cùng tập ảnh, cùng thứ tự file)."""
    shapes = {np.shape(p) for p in list_of_probs}
    if len(shapes) != 1:
        raise ValueError(f"các mô hình có kích thước dự đoán khác nhau: {shapes}")
    p = np.mean(list_of_probs, axis=0)
    return p / p.sum(1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """T > 0 cực tiểu NLL trên VAL. Tối ưu log T bằng LBFGS (float64), khởi tạo từ tìm lưới thô."""
    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    grid = np.linspace(-2.5, 2.5, 51)
    nll = [F.cross_entropy(z / np.exp(g), y).item() for g in grid]
    log_t = torch.tensor([grid[int(np.argmin(nll))]], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits, T: float):
    """softmax(logits / T)."""
    return _softmax(np.asarray(logits, dtype=np.float64) / T)


def _fuse_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """w' = gamma * w / sqrt(var + eps);  b' = beta + gamma * (b - mean) / sqrt(var + eps)."""
    fused = copy.deepcopy(conv)
    w = conv.weight.detach().double()
    b = conv.bias.detach().double() if conv.bias is not None else torch.zeros(w.shape[0], dtype=torch.float64,
                                                                              device=w.device)
    std = torch.sqrt(bn.running_var.double() + bn.eps)
    gamma = bn.weight.detach().double() if bn.affine else torch.ones_like(std)
    beta = bn.bias.detach().double() if bn.affine else torch.zeros_like(std)
    scale = gamma / std
    fused.weight = nn.Parameter((w * scale.reshape(-1, 1, 1, 1)).to(conv.weight.dtype))
    fused.bias = nn.Parameter((beta + scale * (b - bn.running_mean.double())).to(conv.weight.dtype))
    return fused


def _bn_replacement(bn: nn.Module) -> nn.Module:
    """BN thường -> Identity. BatchNormAct2d của timm (BN + drop + act) -> giữ lại drop + act."""
    act, drop = getattr(bn, "act", None), getattr(bn, "drop", None)
    if act is None and drop is None:
        return nn.Identity()
    return nn.Sequential(drop or nn.Identity(), act or nn.Identity())


def fuse_conv_bn(model):
    """Gộp mọi cặp (Conv2d, BatchNorm2d) liền kề trong cùng module cha. Trả về bản sao đã gộp.

    "Liền kề" theo thứ tự khai báo con; với timm ResNet/EfficientNet/MobileNetV3 thứ tự này trùng
    luồng dữ liệu. Luôn kiểm tra lại bằng check_fusion (sai số đầu ra).
    """
    fused = copy.deepcopy(model).eval()
    n_fused = 0
    for parent in list(fused.modules()):
        children = list(parent.named_children())
        for (n1, m1), (n2, m2) in zip(children, children[1:]):
            if isinstance(m1, nn.Conv2d) and isinstance(m2, nn.BatchNorm2d) and m2.track_running_stats:
                setattr(parent, n1, _fuse_pair(m1, m2))
                setattr(parent, n2, _bn_replacement(m2))
                n_fused += 1
    fused.n_fused_bn = n_fused
    return fused


def count_bn(model) -> int:
    return sum(isinstance(m, nn.BatchNorm2d) for m in model.modules())


def check_fusion(model, fused, img_size: int = 224, n: int = 4, device=None) -> float:
    """Sai số tuyệt đối lớn nhất giữa đầu ra trước và sau khi gộp BN (FP32)."""
    device = device or next(model.parameters()).device
    x = torch.randn(n, 3, img_size, img_size, device=device)
    model.eval()
    fused.eval()
    with torch.inference_mode():
        return float((model(x).float() - fused(x).float()).abs().max())
