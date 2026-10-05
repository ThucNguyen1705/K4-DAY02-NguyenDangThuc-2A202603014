"""pipeline_checks.py - Bước 0: EDA và các kiểm tra pipeline trước khi chạy thật.

    eda_plots(...)            biểu đồ phân bố lớp, ảnh mẫu mỗi lớp, thống kê kích thước/kênh
    initial_loss(...)         loss CE ban đầu của head mới (kỳ vọng ~ ln 9 = 2.197)
    overfit_small_batch(...)  train trên 32 ảnh, loss phải xuống gần 0
    show_augmented(...)       ảnh sau augmentation đã giải chuẩn hoá, kèm nhãn
    check_modes(...)          model.train()/eval() đúng lúc
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import CLASS_NAMES, build_transforms, denormalize, make_loader
from model import build_model, set_train_mode, param_groups


def eda_plots(split_info: dict, train_df, images_dir, fig_dir, n_per_class: int = 4, seed: int = 0) -> dict:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pc: pd.DataFrame = split_info["per_class"]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(pc))
    bottom = np.zeros(len(pc))
    for part in ("train", "val", "test"):
        ax.bar(x, pc[part], bottom=bottom, label=part)
        bottom += pc[part].to_numpy()
    for i, v in enumerate(pc["total"]):
        ax.text(i, v + 80, str(v), ha="center", fontsize=8)
    ax.set_xticks(x, pc.index, rotation=30, ha="right")
    ax.set_ylabel("số ảnh")
    ax.set_title("DeepWeeds fold 0: phân bố lớp theo tập")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_class_distribution.png", dpi=120)
    plt.close(fig)

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(CLASS_NAMES), n_per_class, figsize=(2.2 * n_per_class, 2.2 * len(CLASS_NAMES)))
    for c, name in enumerate(CLASS_NAMES):
        files = train_df.loc[train_df["Label"] == c, "Filename"].to_numpy()
        for j, f in enumerate(rng.choice(files, n_per_class, replace=False)):
            with Image.open(Path(images_dir) / f) as im:
                axes[c, j].imshow(im.convert("RGB"))
            axes[c, j].set_xticks([])
            axes[c, j].set_yticks([])
            if j == 0:
                axes[c, j].set_ylabel(name, fontsize=9)
    fig.suptitle("Ảnh mẫu mỗi lớp (tập train)")
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_samples.png", dpi=100)
    plt.close(fig)

    ratio = pc["total"].max() / pc["total"].min()
    return {"max_min_ratio": float(ratio),
            "largest": pc["total"].idxmax(), "smallest": pc["total"].idxmin(),
            "diff_vs_paper": (pc["total"] - pc["paper_table1"]).to_dict()}


def _tiny_loader(df, images_dir, n, aug_train=False, batch_size=32, seed=0):
    k = max(1, math.ceil(n / 9))
    parts = [df[df["Label"] == c].sample(k, random_state=seed) for c in range(9)]
    sub = pd.concat(parts).head(n).reset_index(drop=True)
    tf = build_transforms(aug_train, 224) if aug_train else build_transforms(False, 224)
    return make_loader(sub, images_dir, tf, batch_size, train=False, num_workers=0)


def initial_loss(backbone: str, val_df, images_dir, device, n: int = 256) -> dict:
    """CE trên vài batch val ngay sau khi thay head mới (chưa train)."""
    torch.manual_seed(0)
    model = build_model(backbone, pretrained=True).to(device).eval()
    loader = _tiny_loader(val_df, images_dir, n, batch_size=64)
    losses, logits_std = [], []
    with torch.inference_mode():
        for x, y, _ in loader:
            out = model(x.to(device)).float()
            losses.append(F.cross_entropy(out, y.to(device), reduction="sum").item())
            logits_std.append(out.std().item())
    loss = sum(losses) / len(loader.dataset)
    return {"backbone": backbone, "initial_loss": loss, "expected": math.log(9),
            "logit_std": float(np.mean(logits_std))}


def overfit_small_batch(backbone: str, train_df, images_dir, device, n: int = 32, steps: int = 150,
                        fig_path=None) -> dict:
    """Train toàn bộ model trên đúng n ảnh cố định (không augmentation). Loss phải về gần 0."""
    torch.manual_seed(0)
    model = build_model(backbone, pretrained=True).to(device)
    loader = _tiny_loader(train_df, images_dir, n, batch_size=n)
    x, y, _ = next(iter(loader))
    x, y = x.to(device), y.to(device)
    opt = torch.optim.AdamW(param_groups(model, 1e-4, 1e-3, 0.0))
    hist = []
    for _ in range(steps):
        set_train_mode(model)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item())
    model.eval()
    with torch.inference_mode():
        acc = (model(x).argmax(1) == y).float().mean().item()
    if fig_path:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.semilogy(hist)
        ax.set_xlabel("bước")
        ax.set_ylabel("train loss (log)")
        ax.set_title(f"Overfit {n} ảnh - {backbone}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
    return {"backbone": backbone, "n_images": n, "steps": steps, "loss_first": hist[0],
            "loss_last": hist[-1], "train_acc_eval_mode": acc}


def show_augmented(train_df, images_dir, fig_path, aug: str = "basic", n: int = 16, seed: int = 0) -> None:
    """Ảnh sau augmentation (đã giải chuẩn hoá) cùng nhãn, để chắc ảnh và nhãn khớp nhau."""
    torch.manual_seed(seed)
    tf = build_transforms(True, 224, aug)
    sub = train_df.sample(n, random_state=seed).reset_index(drop=True)
    loader = make_loader(sub, images_dir, tf, n, train=False, num_workers=0)
    x, y, names = next(iter(loader))
    x = denormalize(x)
    cols = 8
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.5 * rows))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(x[i].permute(1, 2, 0).numpy())
            ax.set_title(f"{CLASS_NAMES[int(y[i])]}\n{names[i][:14]}", fontsize=7)
    fig.suptitle(f"Ảnh train sau augmentation '{aug}' (đã giải chuẩn hoá)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=100)
    plt.close(fig)


def check_modes(device) -> dict:
    """train_one_epoch đặt train mode; evaluate đặt eval mode; đóng băng thì BN backbone ở eval."""
    from train import evaluate
    net = build_model("resnet18", pretrained=False).to(device)
    set_train_mode(net)
    ok_train = net.training and net.bn1.training
    ds = [(torch.randn(3, 64, 64), 0, "a.jpg")] * 4
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    evaluate(net, loader, device=device)
    ok_eval = not any(m.training for m in net.modules())
    frozen = build_model("resnet18", pretrained=False, init="frozen").to(device)
    set_train_mode(frozen)
    ok_frozen = (not frozen.bn1.training) and frozen.get_classifier().training
    return {"train_mode_ok": ok_train, "eval_mode_ok": ok_eval, "frozen_bn_eval_ok": ok_frozen}
