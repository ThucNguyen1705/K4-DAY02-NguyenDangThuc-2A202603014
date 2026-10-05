"""bonus.py - phần làm thêm (RUBRIC mục 2).

    shift_eval(lab)     : lệch phân phối tự tạo trên VAL (mờ, tối, nhiễu, JPEG nén mạnh), macro-F1 và
                          ECE trước/sau temperature scaling (T khớp trên val sạch). Không dùng test.
    gradcam_errors(lab) : Grad-CAM cho các ảnh test bị đoán sai của hai lớp khó (F01 seed 0).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms as T

import inference as inf
from dataset import CLASS_NAMES, load_split, make_loader
from experiments import FULL, S, Lab, _mean_std, recipe_cfg
from train import compute_metrics, load_trained, softmax_np


class _PilOp:
    def __init__(self, kind: str):
        self.kind = kind

    def __call__(self, img):
        if self.kind == "blur":
            return img.filter(ImageFilter.GaussianBlur(radius=2.5))
        if self.kind == "dark":
            return ImageEnhance.Brightness(img).enhance(0.4)
        if self.kind == "jpeg":
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=10)
            return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        return img


class _Noise:
    def __init__(self, std: float, seed: int = 0):
        self.std = std
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self, x):
        return (x + torch.randn(x.shape, generator=self.g) * self.std).clamp(0, 1)


SHIFTS = {"sạch": None, "mờ (Gaussian r=2.5)": "blur", "tối (độ sáng x0.4)": "dark",
          "nhiễu Gaussian std=0.08": "noise", "JPEG quality 10": "jpeg"}


def shift_eval(lab: Lab) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = recipe_cfg(lab, exp_id="F01", seed=0)
    model = load_trained(cfg, device)
    mean, std = _mean_std(model)
    _, val_df, _ = load_split(lab.labels_dir)
    rows, T_clean = [], None
    for name, kind in SHIFTS.items():
        ops = [T.Resize(FULL), T.CenterCrop(S)]
        if kind in ("blur", "dark", "jpeg"):
            ops.append(_PilOp(kind))
        ops.append(T.ToTensor())
        if kind == "noise":
            ops.append(_Noise(0.08))
        ops.append(T.Normalize(mean, std))
        loader = make_loader(val_df, lab.images_dir, T.Compose(ops), 128, train=False, num_workers=0 if kind == "noise" else lab.num_workers)
        names, y, outs = inf.predict_views(model, loader, device)
        z = outs[0]
        if T_clean is None:
            T_clean = inf.fit_temperature(z, y)
        p0, p1 = softmax_np(z), inf.apply_temperature(z, T_clean)
        m0, m1 = compute_metrics(y, p0.argmax(1), p0), compute_metrics(y, p1.argmax(1), p1)
        rows.append({"lệch phân phối (val)": name, "macro-F1 val": m0["macro_f1"], "top-1 val": m0["top1"],
                     "recall Chinee apple": m0["recall"][0], "recall Snake weed": m0["recall"][7],
                     "ECE trước TS": m0["ece"], "ECE sau TS (T từ val sạch)": m1["ece"], "T": T_clean})
    df = pd.DataFrame(rows)
    df.to_csv(lab.root / "bonus_shift.csv", index=False)
    return df


def _gradcam(model, x, cls: int):
    """Grad-CAM trên đầu ra forward_features (CNN: B,C,H,W; Swin: B,H,W,C). ViT/DeiT không dùng."""
    model.zero_grad()
    with torch.enable_grad():
        feats = model.forward_features(x)
        feats.retain_grad()
        logits = model.forward_head(feats)
        logits[0, cls].backward()
    f, g = feats.detach(), feats.grad.detach()
    if f.ndim == 4 and f.shape[-1] > f.shape[1]:  # NHWC (Swin)
        f, g = f.permute(0, 3, 1, 2), g.permute(0, 3, 1, 2)
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * f).sum(1))[0]
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy()


def gradcam_errors(lab: Lab, n_per_class: int = 4) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = recipe_cfg(lab, exp_id="F01", seed=0)
    if any(t in cfg.backbone for t in ("deit", "vit")):
        print("Grad-CAM bỏ qua với ViT/DeiT")
        return None
    model = load_trained(cfg, device).float().to(memory_format=torch.contiguous_format)
    mean, std = _mean_std(model)
    tf = T.Compose([T.Resize(FULL), T.CenterCrop(S), T.ToTensor(), T.Normalize(mean, std)])
    p = pd.read_csv(lab.root / "predictions" / "F01_seed0_test.csv")
    p["conf"] = p[[f"p{i}" for i in range(9)]].max(1)
    wrong = p[(p["y_true"] != p["y_pred"]) & p["y_true"].isin([0, 7])].sort_values("conf", ascending=False)
    pick = pd.concat([wrong[wrong["y_true"] == c].head(n_per_class) for c in (0, 7)])
    if pick.empty:
        return None
    fig, axes = plt.subplots(2, len(pick), figsize=(2.6 * len(pick), 5.6))
    axes = np.atleast_2d(axes)
    for j, (_, r) in enumerate(pick.iterrows()):
        with Image.open(Path(lab.images_dir) / r["Filename"]) as im:
            img = im.convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        cam = _gradcam(model, x, int(r["y_pred"]))
        show = np.asarray(T.CenterCrop(S)(T.Resize(FULL)(img))) / 255.0
        cam_up = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((S, S), Image.BILINEAR)) / 255.0
        axes[0, j].imshow(show)
        axes[0, j].set_title(f"thật: {CLASS_NAMES[int(r['y_true'])]}\nđoán: {CLASS_NAMES[int(r['y_pred'])]} ({r['conf']:.2f})",
                             fontsize=8)
        axes[1, j].imshow(show)
        axes[1, j].imshow(cam_up, cmap="jet", alpha=0.45)
        axes[1, j].set_title("Grad-CAM lớp dự đoán", fontsize=8)
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle(f"Grad-CAM trên ảnh test bị đoán sai (F01 seed 0, {cfg.backbone})")
    fig.tight_layout()
    out = lab.root / "figures" / "gradcam_errors.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out
