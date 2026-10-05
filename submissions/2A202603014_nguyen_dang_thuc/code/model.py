"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
Thêm: set_train_mode (gọi thay cho model.train(), giữ backbone đóng băng ở eval), weight_tag.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune", drop_path_rate: float = 0.0,
                img_size: int | None = None):
    """init: "scratch" (không tải trọng số), "frozen" (chỉ train head), "finetune" (train toàn bộ).

    img_size chỉ dùng cho ViT/DeiT: timm nội suy lại pos-embed của trọng số 224 về kích thước mới.
    """
    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init không hỗ trợ: {init}")
    kw = {"drop_path_rate": drop_path_rate} if drop_path_rate else {}
    if img_size and any(t in name for t in ("vit", "deit")):
        kw["img_size"] = img_size
    model = timm.create_model(name, pretrained=pretrained and init != "scratch",
                              num_classes=num_classes, drop_rate=drop_rate, **kw)
    model.frozen_backbone = False
    if init == "frozen":
        freeze_backbone(model)
    return model


def weight_tag(model, init: str = "finetune") -> str:
    """Tên đầy đủ của bộ trọng số đã tải, ví dụ resnet50.a1_in1k."""
    if init == "scratch":
        return "none (scratch)"
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    arch, tag = cfg.get("architecture", "?"), cfg.get("tag", "")
    return f"{arch}.{tag}" if tag else arch


def _head_modules(model) -> set:
    return set(model.get_classifier().modules())


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head (model.get_classifier()).

    BatchNorm của backbone phải ở eval để running_mean/var không bị cập nhật. Vì vòng train gọi
    model.train() mỗi epoch, ta dùng set_train_mode() thay cho model.train().
    """
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        p.requires_grad_(id(p) in head_ids)
    model.frozen_backbone = True


def set_train_mode(model) -> None:
    """model.train(), nhưng nếu backbone đóng băng thì mọi module ngoài head về lại eval."""
    model.train()
    if getattr(model, "frozen_backbone", False):
        head = _head_modules(model)
        for m in model.modules():
            if m is not model and m not in head:
                m.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """3 nhóm như slide trang 52:
      1. backbone, ndim > 1         : lr_backbone, weight_decay
      2. backbone norm/bias (ndim<=1) và các token/pos-embed timm khai báo no_weight_decay: lr_backbone, wd 0
      3. head mới                    : lr_head, weight_decay
    Bỏ qua tham số requires_grad == False.
    """
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    skip = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    decay, no_decay, head = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            head.append(p)
        elif p.ndim <= 1 or name in skip or name.rsplit(".", 1)[-1] in skip:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"name": "backbone_decay", "params": decay, "lr": lr_backbone, "weight_decay": weight_decay},
        {"name": "backbone_no_decay", "params": no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"name": "head", "params": head, "lr": lr_head, "weight_decay": weight_decay},
    ]
    return [g for g in groups if g["params"]]


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size, đếm bằng fvcore (fvcore đếm 1 MAC = 1 'flop').

    Nếu fvcore lỗi với kiến trúc nào đó thì tự đếm conv/linear bằng hook (bỏ qua matmul của
    attention nên sẽ thấp hơn một chút với ViT/Swin).
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(model, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        macs = fca.total()
    except Exception as e:  # noqa: BLE001
        print(f"fvcore lỗi ({type(e).__name__}), chuyển sang đếm bằng hook")
        macs = _hook_macs(model, x)
    model.train(was_training)
    return macs / 1e9


def _hook_macs(model, x) -> float:
    total = [0]

    def conv_hook(m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        total[0] += out.numel() * k

    def linear_hook(m, inp, out):
        total[0] += out.numel() * m.in_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return float(total[0])
