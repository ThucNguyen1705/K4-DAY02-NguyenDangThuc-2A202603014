"""train.py - vòng huấn luyện dùng chung cho mọi thí nghiệm (B, T, F).

Mọi thí nghiệm đi qua MỘT hàm run(cfg); đổi thí nghiệm bằng cách đổi Config.
Chạy từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Macro-F1 val để chọn checkpoint tính bằng eval.compute_metrics của repo (cùng định nghĩa lúc chấm).

Một số lựa chọn của em (ghi lại để tái lập):
  - LR cập nhật theo từng bước (iteration): warmup tuyến tính rồi cosine về 0.
  - Loss val luôn là CE thường (kể cả khi train bằng focal/LS) để so được giữa các thí nghiệm.
  - Nếu run_dir đã có summary.json với cùng cấu hình thì không train lại (chạy tiếp được khi
    phiên Kaggle bị ngắt). Khi đó nếu bật save_test_predictions mà chưa có file test thì nạp
    checkpoint tốt nhất và chạy test đúng một lần.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np


def _find_eval_dir() -> Path:
    """Tìm thư mục chứa eval.py gốc: biến môi trường LAB_EVAL_DIR, thư mục này, hoặc các thư mục cha."""
    here = Path(__file__).resolve().parent
    cands = [Path(os.environ["LAB_EVAL_DIR"])] if os.environ.get("LAB_EVAL_DIR") else []
    cands += [here, *here.parents]
    for d in cands:
        if (d / "eval.py").exists():
            return d
    raise FileNotFoundError("không tìm thấy eval.py của repo")


sys.path.insert(0, str(_find_eval_dir()))
from eval import compute_metrics, save_predictions  # noqa: E402


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    desc: str = ""                    # mô tả ngắn cho tên ảnh curves/<exp_id>_<desc>.png
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug | vflip
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    optimizer: str = "adamw"          # adamw | sgd
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    channels_last: bool = True
    num_workers: int = 2
    # --- chỉ dùng cho smoke test: lấy n ảnh đầu mỗi lớp của train ---
    limit_train_per_class: int | None = None
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, logit của từng lần chạy
    ckpt_dir: str | None = None       # nơi để best.pt (None = trong run_dir)
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py
    curves_dir: str = "curves"
    measure_latency: bool = False     # đo nhanh độ trễ batch 1 (Bước 1)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


# Các trường không ảnh hưởng kết quả huấn luyện: khác nhau vẫn coi là cùng một lần chạy khi resume.
_RESUME_IGNORE = {"save_test_predictions", "num_workers", "images_dir", "labels_dir", "out_dir",
                  "ckpt_dir", "pred_dir", "curves_dir", "measure_latency", "desc"}


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def ckpt_path(cfg: Config) -> Path:
    base = Path(cfg.ckpt_dir) / cfg.exp_id / f"seed{cfg.seed}" if cfg.ckpt_dir else run_dir(cfg)
    return base / "best.pt"


def curve_path(cfg: Config) -> Path:
    desc = cfg.desc or cfg.backbone
    name = f"{cfg.exp_id}_{desc}.png" if cfg.seed == 0 else f"{cfg.exp_id}_seed{cfg.seed}_{desc}.png"
    return Path(cfg.curves_dir) / name


def set_seed(seed: int) -> None:
    """Cố định random, numpy, torch (CPU, CUDA). Worker của DataLoader được seed trong make_loader.

    cudnn.benchmark = True để chạy nhanh, nên kết quả không trùng từng bit giữa hai lần chạy
    (sai khác nhỏ do thuật toán tích chập không tất định). Mức tái lập này được ghi trong báo cáo.
    """
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_optimizer(model, cfg: Config):
    """AdamW (hoặc SGD momentum) với 3 nhóm tham số của model.param_groups."""
    import torch
    from model import param_groups
    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    raise ValueError(f"optimizer không hỗ trợ: {cfg.optimizer}")


def lr_factor(step: int, total: int, warmup: int) -> float:
    """Hệ số nhân LR tại bước `step`: warmup tuyến tính rồi cosine về 0."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """LambdaLR cập nhật theo bước. Giữ nguyên tỉ lệ LR head / backbone giữa các nhóm."""
    import torch
    total = cfg.epochs * steps_per_epoch
    warmup = int(round(cfg.warmup_epochs * steps_per_epoch))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, total, warmup))


class EMA:
    """W_ema <- d * W_ema + (1 - d) * W sau mỗi bước tối ưu.

    - Bản sao riêng (self.module) dùng để đánh giá.
    - Buffer số thực của BatchNorm (running_mean/var) cũng lấy trung bình động như tham số,
      giống ModelEmaV2 của timm; buffer số nguyên (num_batches_tracked) thì chép thẳng.
    - Decay có warmup d_t = min(d, (1 + t) / (10 + t)) để đầu quá trình EMA không bị kéo về
      trọng số ban đầu (head ngẫu nhiên).
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.steps = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model) -> None:
        import torch
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        with torch.no_grad():
            src = model.state_dict()
            for k, v in self.module.state_dict().items():
                s = src[k].detach()
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(s, alpha=1 - d)
                else:
                    v.copy_(s)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch. Trả về {"train_loss", "train_acc" (NaN nếu có mix), "lr", "lrs"}."""
    import torch
    from losses import mix_batch, mixed_loss
    from model import set_train_mode

    set_train_mode(model)
    use_amp = cfg.amp and device.type == "cuda"
    tot_loss, tot_correct, tot_n, lrs = 0.0, 0, 0, []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        targets = None
        if cfg.mix:
            x, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        loss = mixed_loss(criterion, logits, targets) if targets else criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss = {loss.item()} (không hữu hạn)")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        bs = y.size(0)
        tot_loss += loss.item() * bs
        tot_n += bs
        if targets is None:
            tot_correct += (logits.argmax(1) == y).sum().item()
        lrs.append(optimizer.param_groups[0]["lr"])
    return {"train_loss": tot_loss / tot_n,
            "train_acc": tot_correct / tot_n if not cfg.mix else float("nan"),
            "lr": lrs[-1], "lrs": lrs}


def evaluate(model, loader, criterion=None, device=None, amp: bool = True):
    """Chạy model ở chế độ eval, không gradient, giữ thứ tự loader.

    Trả về (filenames, y_true[N], logits[N, 9], loss). criterion=None thì loss là CE thường.
    """
    import torch
    import torch.nn.functional as F
    device = device or next(model.parameters()).device
    model.eval()
    names, ys, outs = [], [], []
    use_amp = amp and device.type == "cuda"
    with torch.inference_mode():
        for x, y, fn in loader:
            x = x.to(device, non_blocking=True)
            if x.dim() == 4 and device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
            outs.append(out.float().cpu())
            ys.append(y)
            names.extend(fn)
    logits = torch.cat(outs)
    y_true = torch.cat(ys)
    crit = criterion or F.cross_entropy
    loss = float(crit(logits, y_true))
    return names, y_true.numpy(), logits.numpy(), loss


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def plot_curves(history: list[dict], path: str | Path, title: str, lrs: list[float] | None = None) -> None:
    """Loss train/val, macro-F1/top-1 val (và acc train), LR theo bước."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = {k: [r[k] for r in history] for k in history[0]}
    ep = h["epoch"]
    ncol = 3 if lrs else 2
    fig, ax = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4))
    ax[0].plot(ep, h["train_loss"], "o-", label="train loss")
    ax[0].plot(ep, h["val_loss"], "s-", label="val loss (CE)")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(ep, h["val_macro_f1"], "o-", label="val macro-F1")
    ax[1].plot(ep, h["val_top1"], "s-", label="val top-1")
    if not all(math.isnan(v) for v in h["train_acc"]):
        ax[1].plot(ep, h["train_acc"], "^--", label="train top-1")
    if "val_macro_f1_raw" in h:
        ax[1].plot(ep, h["val_macro_f1_raw"], "x:", label="val macro-F1 (không EMA)")
    best = int(np.argmax(h["val_macro_f1"]))
    ax[1].axvline(ep[best], color="gray", ls="--", lw=1, label=f"best epoch {ep[best]}")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("metric")
    ax[1].legend(loc="lower right")
    ax[1].grid(alpha=0.3)

    if lrs:
        ax[2].plot(np.arange(1, len(lrs) + 1), lrs)
        ax[2].set_xlabel("bước (iteration)")
        ax[2].set_ylabel("LR backbone")
        ax[2].set_title("warmup + cosine")
        ax[2].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _same_config(saved: dict, cfg: Config) -> bool:
    cur = asdict(cfg)
    return all(saved.get(k) == v for k, v in cur.items() if k not in _RESUME_IGNORE)


def _subset_train(df, n):
    return df.groupby("Label", group_keys=False).head(n).reset_index(drop=True)


def _eval_transforms_for(model, cfg: Config):
    import timm.data
    from dataset import build_transforms
    dc = timm.data.resolve_data_config({}, model=model)
    return dc["mean"], dc["std"], build_transforms(False, cfg.img_size, mean=dc["mean"], std=dc["std"])


def _save_split_outputs(cfg: Config, rd: Path, split: str, names, y, logits) -> None:
    np.save(rd / f"{split}_logits.npy", logits.astype(np.float32))
    np.save(rd / f"{split}_labels.npy", y.astype(np.int64))
    (rd / f"{split}_filenames.json").write_text(json.dumps(list(names)))
    save_predictions(pred_path(cfg, split), names, y, softmax_np(logits))


def _run_test_once(cfg: Config, model, test_df, device, rd: Path) -> dict:
    """Bước 4: chạy test đúng một lần với checkpoint tốt nhất (đã chọn trên val)."""
    from dataset import make_loader
    if pred_path(cfg, "test").exists():
        raise RuntimeError(f"{pred_path(cfg, 'test')} đã có: test chỉ chạy một lần mỗi seed")
    _, _, tf = _eval_transforms_for(model, cfg)
    loader = make_loader(test_df, cfg.images_dir, tf, cfg.batch_size * 2, train=False,
                         num_workers=cfg.num_workers)
    names, y, logits, _ = evaluate(model, loader, device=device, amp=cfg.amp)
    _save_split_outputs(cfg, rd, "test", names, y, logits)
    p = softmax_np(logits)
    m = compute_metrics(y, p.argmax(1), p)
    return {"test_macro_f1": m["macro_f1"], "test_top1": m["top1"], "test_ece": m["ece"]}


def load_trained(cfg: Config, device=None):
    """Nạp lại model với checkpoint tốt nhất của một lần chạy (dùng ở Bước 3, 4)."""
    import torch
    from model import build_model
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.backbone, pretrained=False, drop_rate=cfg.drop_rate, init="finetune",
                        drop_path_rate=cfg.drop_path_rate, img_size=cfg.img_size)
    model.load_state_dict(torch.load(ckpt_path(cfg), map_location="cpu"))
    model.to(device).eval()
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict tóm tắt."""
    import torch
    import timm
    from dataset import build_transforms, check_split, load_split, make_loader, preload_images
    from losses import build_criterion, class_weights
    from model import build_model, count_gmacs, count_params, weight_tag

    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = load_split(cfg.labels_dir, cfg.fold)
    check_split(train_df, val_df, test_df, cfg.images_dir, verbose=False)

    # --- chạy tiếp: cùng cấu hình đã xong thì không train lại ---
    summ_file = rd / "summary.json"
    if summ_file.exists():
        saved = json.loads((rd / "config.json").read_text())
        summary = json.loads(summ_file.read_text())
        if _same_config(saved, cfg):
            if cfg.save_test_predictions and "test_macro_f1" not in summary:
                model = load_trained(cfg, device)
                preload_images(test_df["Filename"], cfg.images_dir)
                summary.update(_run_test_once(cfg, model, test_df, device, rd))
                summ_file.write_text(json.dumps(summary, indent=2))
            print(f"[{cfg.exp_id} seed{cfg.seed}] đã có kết quả, bỏ qua huấn luyện")
            return summary
        print(f"[{cfg.exp_id} seed{cfg.seed}] cấu hình khác lần trước, huấn luyện lại")

    set_seed(cfg.seed)
    (rd / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    if cfg.limit_train_per_class:
        train_df = _subset_train(train_df, cfg.limit_train_per_class)
    preload_images(pd_concat_names(train_df, val_df), cfg.images_dir)

    model = build_model(cfg.backbone, pretrained=True, drop_rate=cfg.drop_rate, init=cfg.init,
                        drop_path_rate=cfg.drop_path_rate, img_size=cfg.img_size).to(device)
    tag = weight_tag(model, cfg.init)
    params_m = count_params(model)
    gmacs = count_gmacs(model, cfg.img_size)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    dc = timm.data.resolve_data_config({}, model=model)
    mean, std = dc["mean"], dc["std"]

    train_tf = build_transforms(True, cfg.img_size, cfg.aug, mean=mean, std=std)
    eval_tf = build_transforms(False, cfg.img_size, mean=mean, std=std)
    train_loader = make_loader(train_df, cfg.images_dir, train_tf, cfg.batch_size, train=True,
                               sampler=cfg.sampler, num_workers=cfg.num_workers, seed=cfg.seed)
    val_loader = make_loader(val_df, cfg.images_dir, eval_tf, cfg.batch_size * 2, train=False,
                             num_workers=cfg.num_workers)

    weight = None
    if cfg.loss == "ce_weighted":
        counts = np.bincount(train_df["Label"].astype(int), minlength=9)  # chỉ dùng train
        weight = class_weights(counts, cfg.class_weight_beta or 0.0).to(device)
    criterion = build_criterion(cfg.loss, smoothing=cfg.label_smoothing or 0.1,
                                gamma=cfg.focal_gamma, weight=weight).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    ck = ckpt_path(cfg)
    ck.parent.mkdir(parents=True, exist_ok=True)
    history, all_lrs, epoch_times = [], [], []
    best_f1, best_epoch = -1.0, -1
    print(f"[{cfg.exp_id} seed{cfg.seed}] {cfg.backbone} ({tag}) {params_m:.1f}M, {gmacs:.2f} GMAC, "
          f"{len(train_df)} ảnh train, {len(train_loader)} bước/epoch, mean={mean}")
    for epoch in range(1, cfg.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t0)
        all_lrs.extend(tr.pop("lrs"))

        eval_model = ema.module if ema else model
        names, y, logits, vloss = evaluate(eval_model, val_loader, device=device, amp=cfg.amp)
        p = softmax_np(logits)
        m = compute_metrics(y, p.argmax(1), p)
        row = {"epoch": epoch, **tr, "val_loss": vloss, "val_top1": m["top1"],
               "val_macro_f1": m["macro_f1"], "val_bal_acc": m["balanced_acc"], "val_ece": m["ece"],
               "epoch_time_s": epoch_times[-1]}
        raw = None
        if ema:  # so EMA với trọng số thường trong cùng lần chạy (I06)
            raw = evaluate(model, val_loader, device=device, amp=cfg.amp)
            pr = softmax_np(raw[2])
            row["val_macro_f1_raw"] = compute_metrics(raw[1], pr.argmax(1), pr)["macro_f1"]
        history.append(row)

        if m["macro_f1"] > best_f1:  # dấu > nên hoà thì giữ epoch sớm hơn
            best_f1, best_epoch = m["macro_f1"], epoch
            torch.save(eval_model.state_dict(), ck)
            if raw is not None:
                np.save(rd / "val_logits_raw.npy", raw[2].astype(np.float32))
        pd_write(history, rd / "history.csv")
        print(f"  ep {epoch:2d} | loss {tr['train_loss']:.4f} | val loss {vloss:.4f} | "
              f"val F1 {m['macro_f1']:.4f} | val top1 {m['top1']:.4f} | {epoch_times[-1]:.0f}s")

    # --- nạp checkpoint tốt nhất, lưu logit + file dự đoán val ---
    eval_model = ema.module if ema else model
    eval_model.load_state_dict(torch.load(ck, map_location=device))
    names, y, logits, _ = evaluate(eval_model, val_loader, device=device, amp=cfg.amp)
    _save_split_outputs(cfg, rd, "val", names, y, logits)
    p = softmax_np(logits)
    m = compute_metrics(y, p.argmax(1), p)
    np.save(rd / "lrs.npy", np.asarray(all_lrs, dtype=np.float32))

    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "weight_tag": tag,
        "params_m": params_m, "gmacs": gmacs, "img_size": cfg.img_size, "epochs": cfg.epochs,
        "best_epoch": best_epoch, "val_macro_f1": m["macro_f1"], "val_top1": m["top1"],
        "val_bal_acc": m["balanced_acc"], "val_ece": m["ece"], "val_nll": m["nll"],
        "val_f1_per_class": m["f1"].tolist(), "val_recall_per_class": m["recall"].tolist(),
        "sec_per_epoch": float(np.mean(epoch_times)), "train_time_s": float(np.sum(epoch_times)),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__, "timm": timm.__version__,
    }
    if ema:
        summary["val_macro_f1_raw_at_best"] = history[best_epoch - 1]["val_macro_f1_raw"]
    if cfg.measure_latency and device.type == "cuda":
        from benchmark import latency_report
        lat = latency_report(eval_model, 1, cfg.img_size, "fp32", "cuda", warmup=10, iters=50)
        summary["latency_b1_p50_ms"] = lat["p50"]
        summary["latency_b1_p95_ms"] = lat["p95"]
    if cfg.save_test_predictions:
        preload_images(test_df["Filename"], cfg.images_dir)
        summary.update(_run_test_once(cfg, eval_model, test_df, device, rd))

    title = f"{cfg.exp_id} seed{cfg.seed} | {cfg.backbone} | {cfg.desc or 'T00 recipe'}"
    plot_curves(history, curve_path(cfg), title, all_lrs)
    summ_file.write_text(json.dumps(summary, indent=2))
    print(f"[{cfg.exp_id} seed{cfg.seed}] xong: best epoch {best_epoch}, val macro-F1 {m['macro_f1']:.4f}, "
          f"val top-1 {m['top1']:.4f}, {summary['sec_per_epoch']:.0f}s/epoch")
    del model, eval_model, ema, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def pd_concat_names(*dfs):
    return [f for df in dfs for f in df["Filename"]]


def pd_write(history: list[dict], path: Path) -> None:
    import pandas as pd
    pd.DataFrame(history).to_csv(path, index=False)


def _convert(value: str, type_str: str):
    t = str(type_str).replace(" ", "")
    optional = "None" in t
    base = t.replace("|None", "").replace("None|", "")
    if optional and value.lower() in ("none", "null", ""):
        return None
    if base == "bool":
        if value.lower() in ("1", "true", "yes", "y"):
            return True
        if value.lower() in ("0", "false", "no", "n"):
            return False
        raise ValueError(f"giá trị bool không hợp lệ: {value}")
    if base == "int":
        return int(value)
    if base == "float":
        return float(value)
    return value


def parse_overrides(pairs: list[str]) -> dict:
    """['seed=1', 'loss=focal', 'ema_decay=none'] -> dict đã ép kiểu theo field của Config."""
    types = {f.name: f.type for f in fields(Config)}
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"'{pair}' không có dạng KEY=VALUE")
        key, value = pair.split("=", 1)
        key = key.strip()
        if key not in types:
            raise KeyError(f"Config không có trường '{key}'. Các trường: {sorted(types)}")
        out[key] = _convert(value.strip(), types[key])
    return out


def main() -> None:
    """python train.py --set exp_id=B01 backbone=resnet50 seed=0"""
    ap = argparse.ArgumentParser(description="Huấn luyện một cấu hình DeepWeeds")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    res = run(cfg)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
