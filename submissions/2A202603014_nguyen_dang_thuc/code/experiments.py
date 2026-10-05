"""experiments.py - danh sách thí nghiệm và các bước 1-4. Notebook chỉ gọi các hàm ở đây.

Mọi quyết định (chọn backbone, chọn công thức, chọn cách suy luận) chỉ dựa trên VAL và được ghi
ra file decisions.json cùng lý do. Test chỉ được chạy ở step4, mỗi seed một lần.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import inference as inf
from benchmark import latency_report, latency_with_preprocess, multi_forward_latency
from dataset import CLASS_NAMES, build_transforms, load_split, make_loader, preload_images
from train import Config, compute_metrics, load_trained, run, run_dir, save_predictions, softmax_np

HARD = [0, 7]  # Chinee Apple, Snake Weed

BACKBONES = [  # exp_id, tên timm, mô tả ngắn (tên ảnh curves)
    ("B01", "resnet50", "resnet50"),
    ("B02", "convnext_tiny", "convnext_tiny"),
    ("B03", "deit_small_patch16_224", "deit_small"),
    ("B04", "efficientnet_b0", "efficientnet_b0"),
    ("B05", "mobilenetv3_large_100", "mobilenetv3_large"),
]

# exp_id, trục, khác T00 ở điểm nào, thay đổi trong Config, mô tả ngắn
VARIANTS = [
    ("T01", "A", "init=frozen (chỉ train head)", {"init": "frozen"}, "frozen"),
    ("T02", "B", "aug=trivial (+TrivialAugmentWide)", {"aug": "trivial"}, "trivialaug"),
    ("T03", "B", "mix=cutmix (alpha=1)", {"mix": "cutmix"}, "cutmix"),
    ("T04", "C", "loss=label smoothing eps=0.1", {"loss": "ls", "label_smoothing": 0.1}, "labelsmooth"),
    ("T05", "C", "loss=CE trọng số 1/n_c", {"loss": "ce_weighted", "class_weight_beta": 0.0}, "ce_weighted"),
    ("T06", "F", "EMA decay=0.99", {"ema_decay": 0.99}, "ema"),
]
EMA_RUN = "T06"
REBALANCE = {"T05"}  # cách cân bằng lớp (nếu thêm sampler thì không gộp cùng lúc)

# Ngân sách GPU (~30 phút cho cả notebook trên một T4): giảm so với công thức nền của GUIDE
# theo đúng thứ tự gợi ý ở GUIDE mục 7: ít epoch hơn, hạ độ phân giải, ablation trên 1 backbone, 1 seed.
FAST = dict(img_size=128, epochs=4, batch_size=128, warmup_epochs=0.5)


@dataclass
class Lab:
    images_dir: str
    labels_dir: str
    out: str                      # thư mục gốc cho runs/, curves/, predictions/, figures/, ...
    ckpt_dir: str
    num_workers: int = 4
    smoke: bool = False

    @property
    def root(self) -> Path:
        return Path(self.out)

    def cfg(self, **kw) -> Config:
        base = dict(images_dir=self.images_dir, labels_dir=self.labels_dir,
                    out_dir=str(self.root / "runs"), pred_dir=str(self.root / "predictions"),
                    curves_dir=str(self.root / "curves"), ckpt_dir=self.ckpt_dir,
                    num_workers=self.num_workers, **FAST)
        if self.smoke:
            base.update(epochs=1, limit_train_per_class=24, warmup_epochs=0.5)
        base.update(kw)
        return Config(**base)

    def save_json(self, name: str, obj) -> None:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float))

    def load_json(self, name: str):
        return json.loads((self.root / name).read_text())

    def decide(self, key: str, value) -> None:
        p = self.root / "decisions.json"
        d = json.loads(p.read_text()) if p.exists() else {}
        d[key] = value
        p.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=float))


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _val_logits(cfg: Config, name: str = "val_logits.npy"):
    rd = run_dir(cfg)
    return (np.load(rd / name), np.load(rd / "val_labels.npy"),
            json.loads((rd / "val_filenames.json").read_text()))


def _metrics_row(y, probs) -> dict:
    m = compute_metrics(y, probs.argmax(1), probs)
    return {"val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_ece": m["ece"], "val_nll": m["nll"],
            "f1_chinee": m["f1"][0], "f1_snake": m["f1"][7]}


# ----------------------------------------------------------------------------- #
# Bước 1
# ----------------------------------------------------------------------------- #
def step1(lab: Lab) -> pd.DataFrame:
    rows = []
    for exp_id, name, desc in BACKBONES:
        s = run(lab.cfg(exp_id=exp_id, backbone=name, desc=desc))
        rows.append(s)
    df = pd.DataFrame(rows)
    df.to_csv(lab.root / "step1_backbones.csv", index=False)
    best = df.sort_values(["val_macro_f1", "params_m"], ascending=[False, True]).iloc[0]
    lab.decide("step1_backbone", {
        "exp_id": best["exp_id"], "backbone": best["backbone"],
        "reason": f"macro-F1 val cao nhất ({best['val_macro_f1']:.4f}) trong {len(df)} backbone, 1 seed"})
    return df


# ----------------------------------------------------------------------------- #
# Bước 2
# ----------------------------------------------------------------------------- #
def alias_run(src: Config, dst: Config) -> dict:
    """T00 seed 0 trùng cấu hình với lần chạy B0x của cùng backbone (chỉ khác exp_id), nên chép lại
    log, checkpoint, ảnh đường cong và file dự đoán val thay vì train lại. config.json ghi rõ nguồn."""
    import shutil
    from dataclasses import asdict
    from train import ckpt_path, curve_path, pred_path
    sd, dd = run_dir(src), run_dir(dst)
    if (dd / "summary.json").exists():
        return json.loads((dd / "summary.json").read_text())
    dd.mkdir(parents=True, exist_ok=True)
    for f in sd.iterdir():
        if f.is_file() and f.name not in ("config.json", "summary.json"):
            shutil.copy2(f, dd / f.name)
    ckpt_path(dst).parent.mkdir(parents=True, exist_ok=True)
    if ckpt_path(src).exists() and ckpt_path(src) != ckpt_path(dst):
        shutil.copy2(ckpt_path(src), ckpt_path(dst))
    shutil.copy2(curve_path(src), curve_path(dst))
    shutil.copy2(pred_path(src, "val"), pred_path(dst, "val"))
    (dd / "config.json").write_text(json.dumps({**asdict(dst), "copied_from": src.exp_id}, indent=2))
    summary = {**json.loads((sd / "summary.json").read_text()), "exp_id": dst.exp_id, "copied_from": src.exp_id}
    (dd / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def step2(lab: Lab, backbone: str) -> pd.DataFrame:
    base = dict(backbone=backbone)
    rows = []
    b_src = next(e for e, name, _ in BACKBONES if name == backbone)
    b_desc = next(d for e, name, d in BACKBONES if name == backbone)
    for seed in (0, 1, 2):  # đo nhiễu do seed của T00 (chỉ val)
        cfg = lab.cfg(exp_id="T00", seed=seed, desc="baseline", **base)
        if seed == 0:
            s = alias_run(lab.cfg(exp_id=b_src, backbone=backbone, desc=b_desc), cfg)
        else:
            s = run(cfg)
        rows.append({**s, "axis": "-", "change": "công thức nền T00", "cfg": {}})
    noise_std = float(np.std([r["val_macro_f1"] for r in rows], ddof=1))
    t00 = rows[0]["val_macro_f1"]
    for exp_id, axis, change, kw, desc in VARIANTS:
        s = run(lab.cfg(exp_id=exp_id, desc=desc, **base, **kw))
        rows.append({**s, "axis": axis, "change": change, "cfg": kw})

    df = pd.DataFrame(rows)
    df["delta_vs_T00"] = df["val_macro_f1"] - t00
    combo_kw, combo_ids, note = _choose_combination(df, noise_std)
    s = run(lab.cfg(exp_id="T07", desc="combo_" + "+".join(combo_ids), **base, **combo_kw))
    combo_row = {**s, "axis": "kết hợp", "change": "kết hợp " + " + ".join(combo_ids), "cfg": combo_kw,
                 "delta_vs_T00": s["val_macro_f1"] - t00}
    df = pd.concat([df, pd.DataFrame([combo_row])], ignore_index=True)
    df["noise_std"] = noise_std
    df["distinguishable"] = df["delta_vs_T00"].abs() > noise_std
    df.drop(columns=["cfg"]).to_csv(lab.root / "step2_training.csv", index=False)

    cand = df[(df["seed"] == 0)].sort_values("val_macro_f1", ascending=False).iloc[0]
    recipe = {} if cand["exp_id"] == "T00" else dict(df.loc[cand.name, "cfg"])
    lab.decide("step2_noise_std", noise_std)
    lab.decide("step2_combination", {"members": combo_ids, "cfg": combo_kw, "note": note})
    lab.decide("step2_recipe", {
        "exp_id": cand["exp_id"], "cfg": recipe, "backbone": backbone,
        "val_macro_f1": cand["val_macro_f1"], "delta_vs_T00": cand["delta_vs_T00"],
        "reason": (f"macro-F1 val cao nhất ({cand['val_macro_f1']:.4f}) trong các cấu hình seed 0; "
                   f"Δ so với T00 = {cand['delta_vs_T00']:+.4f}, std nhiễu T00 = {noise_std:.4f} "
                   f"({'vượt' if abs(cand['delta_vs_T00']) > noise_std else 'không vượt'} nhiễu)")})
    return df


def _choose_combination(df: pd.DataFrame, noise_std: float):
    """Mỗi trục B-F lấy biến thể tốt nhất có Δ > std nhiễu (tham lam theo trục). Nếu được ít hơn 2
    yếu tố thì bổ sung các biến thể Δ dương lớn nhất ở trục khác để vẫn thử được hiệu ứng cộng dồn."""
    var = df[df["exp_id"].isin([v[0] for v in VARIANTS]) & (df["axis"] != "A")]
    var = var.sort_values("delta_vs_T00", ascending=False)
    chosen, axes_used = [], set()
    for _, r in var.iterrows():
        if r["delta_vs_T00"] > noise_std and r["axis"] not in axes_used:
            if r["exp_id"] in REBALANCE and any(c in REBALANCE for c in chosen):
                continue
            chosen.append(r["exp_id"])
            axes_used.add(r["axis"])
    note = "các yếu tố đều có Δ > std nhiễu"
    if len(chosen) < 2:
        note = "chưa đủ 2 yếu tố vượt nhiễu, bổ sung yếu tố Δ lớn nhất ở trục khác"
        for _, r in var.iterrows():
            if len(chosen) >= 2:
                break
            if r["exp_id"] in chosen or r["axis"] in axes_used:
                continue
            if r["exp_id"] in REBALANCE and any(c in REBALANCE for c in chosen):
                continue
            chosen.append(r["exp_id"])
            axes_used.add(r["axis"])
    kw = {}
    for exp_id in chosen:
        kw.update(next(v[3] for v in VARIANTS if v[0] == exp_id))
    return kw, chosen, note


def recipe_cfg(lab: Lab, **kw) -> Config:
    rec = lab.load_json("decisions.json")["step2_recipe"]
    return lab.cfg(backbone=rec["backbone"], **rec["cfg"], **kw)


# ----------------------------------------------------------------------------- #
# Bước 3
# ----------------------------------------------------------------------------- #
S = FAST["img_size"]               # độ phân giải train/val (128)
FULL = int(round(S / 0.875))        # ảnh đầy đủ resize về 146 để cắt 5 crop 128
METHODS = {  # tên -> (mô tả, K, kiểu loader, hàm tạo view, không gian gộp)
    "I00": (f"1 view (resize {FULL} + center crop {S})", 1, f"crop{S}", None, "prob"),
    "I01": ("TTA lật ngang, gộp xác suất", 2, f"crop{S}", inf.views_hflip, "prob"),
    "I01L": ("TTA lật ngang, gộp logit", 2, f"crop{S}", inf.views_hflip, "logit"),
    "I02": (f"TTA 5 crop {S} từ ảnh {FULL}, gộp xác suất", 5, f"full{FULL}", lambda x: inf.views_multicrop(x, S), "prob"),
    "I02L": (f"TTA 5 crop {S} từ ảnh {FULL}, gộp logit", 5, f"full{FULL}", lambda x: inf.views_multicrop(x, S), "logit"),
    "I02F": ("TTA 10 crop (5 crop + lật), gộp xác suất", 10, f"full{FULL}",
             lambda x: inf.views_multicrop(x, S, flip=True), "prob"),
}
RESOLUTIONS = [160, 192, 224]  # FixRes: kiểm tra ở độ phân giải cao hơn lúc train (128)
TRANSFORMERS = ("deit", "vit", "swin")


def _loader(lab: Lab, df, kind: str, mean, std, bs: int = 128):
    if kind.startswith("full"):
        tf = build_transforms(False, int(kind[4:]), mean=mean, std=std, eval_mode="full")
    else:
        size = int(kind.replace("crop", ""))
        tf = build_transforms(False, size, mean=mean, std=std)
    return make_loader(df, lab.images_dir, tf, bs, train=False, num_workers=lab.num_workers)


def _mean_std(model):
    import timm.data
    dc = timm.data.resolve_data_config({}, model=model)
    return dc["mean"], dc["std"]


def method_scores(model, lab: Lab, df, method: str, device):
    """Trả về (filenames, y, z) với z là 'logit cuối' của phương pháp: logit (1 view / gộp logit)
    hoặc log của xác suất đã gộp. softmax(z / T) dùng chung cho temperature scaling."""
    mean, std = _mean_std(model)
    if method.startswith("R"):  # dò độ phân giải: R288 ...
        names, y, outs = inf.predict_views(model, _loader(lab, df, f"crop{int(method[1:])}", mean, std), device)
        return names, y, outs[0]
    _, _, kind, views, space = METHODS[method]
    names, y, outs = inf.predict_views(model, _loader(lab, df, kind, mean, std), device, views)
    if len(outs) == 1:
        return names, y, outs[0]
    if space == "logit":
        return names, y, np.mean(outs, axis=0)
    return names, y, np.log(np.clip(inf.aggregate_views(outs, "prob"), 1e-12, None))


def _cross_fit_ece(z, y, folds: int = 2, seed: int = 0) -> float:
    """ECE sau TS ước lượng trung thực hơn: khớp T trên một nửa val, đo ECE trên nửa còn lại."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    parts = np.array_split(idx, folds)
    eces = []
    for k in range(folds):
        fit = np.concatenate([p for j, p in enumerate(parts) if j != k])
        T = inf.fit_temperature(z[fit], y[fit])
        p = inf.apply_temperature(z[parts[k]], T)
        eces.append(compute_metrics(y[parts[k]], p.argmax(1), p)["ece"])
    return float(np.mean(eces))


def reliability_plot(y, probs_list, labels, path, bins: int = 15) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="hiệu chuẩn hoàn hảo")
    edges = np.linspace(0, 1, bins + 1)
    for probs, lab_ in zip(probs_list, labels):
        conf, pred = probs.max(1), probs.argmax(1)
        idx = np.clip(np.ceil(conf * bins).astype(int) - 1, 0, bins - 1)
        xs, ys = [], []
        for m in range(bins):
            mask = idx == m
            if mask.sum() >= 5:
                xs.append(conf[mask].mean())
                ys.append((pred[mask] == y[mask]).mean())
        ax.plot(xs, ys, "o-", label=lab_)
    ax.set_xlabel("độ tin cậy (max softmax)")
    ax.set_ylabel("accuracy trong bin")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_title("Reliability diagram (val)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def step3(lab: Lab) -> dict:
    device = _device()
    dec = lab.load_json("decisions.json")
    rec = dec["step2_recipe"]
    noise = dec["step2_noise_std"]
    cfg = recipe_cfg(lab, exp_id=rec["exp_id"])
    model = load_trained(cfg, device)
    _, val_df, _ = load_split(lab.labels_dir)
    preload_images(val_df["Filename"], lab.images_dir)
    is_tf = any(t in cfg.backbone for t in TRANSFORMERS)
    figs = lab.root / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    rows, scores = [], {}
    for meth, (desc, k, *_r) in METHODS.items():
        names, y, z = method_scores(model, lab, val_df, meth, device)
        scores[meth] = z
        rows.append({"exp_id": meth, "method": desc, "model": f"{rec['exp_id']} seed0 ({cfg.backbone})", "K": k,
                     **_metrics_row(y, softmax_np(z))})
    for r in RESOLUTIONS:
        meth = f"R{r}"
        if is_tf:
            rows.append({"exp_id": "I04_" + meth, "method": f"độ phân giải kiểm tra {r}",
                         "model": cfg.backbone, "K": 1, "note": "không áp dụng cho ViT/DeiT (pos-embed cố định theo kích thước train)"})
            continue
        names, y, z = method_scores(model, lab, val_df, meth, device)
        scores[meth] = z
        rows.append({"exp_id": "I04_" + meth, "method": f"độ phân giải kiểm tra {r} (resize {round(r / 0.875)} + crop {r})",
                     "model": f"{rec['exp_id']} seed0 ({cfg.backbone})", "K": 1, **_metrics_row(y, softmax_np(z))})

    # I05 ensemble: (a) 3 backbone tốt nhất Bước 1; (b) T00 ba seed. Chỉ cần logit val đã lưu.
    b1 = pd.read_csv(lab.root / "step1_backbones.csv").sort_values("val_macro_f1", ascending=False)
    top3 = b1.head(3)
    probs = [softmax_np(_val_logits(lab.cfg(exp_id=e))[0]) for e in top3["exp_id"]]
    y0 = _val_logits(lab.cfg(exp_id=top3["exp_id"].iloc[0]))[1]
    rows.append({"exp_id": "I05a", "method": "ensemble 3 backbone tốt nhất Bước 1 (TB xác suất)",
                 "model": " + ".join(f"{e}:{b}" for e, b in zip(top3["exp_id"], top3["backbone"])), "K": 3,
                 **_metrics_row(y0, inf.ensemble_probs(probs))})
    probs = [softmax_np(_val_logits(lab.cfg(exp_id="T00", seed=s))[0]) for s in (0, 1, 2)]
    rows.append({"exp_id": "I05b", "method": "ensemble T00 ba seed (TB xác suất)", "model": "T00 seed0,1,2",
                 "K": 3, **_metrics_row(y0, inf.ensemble_probs(probs))})

    # I06 EMA: cùng lần chạy có EMA, trọng số EMA vs trọng số thường ở cùng epoch
    t11 = lab.cfg(exp_id=EMA_RUN)
    if (run_dir(t11) / "val_logits_raw.npy").exists():
        z_ema, y11, _ = _val_logits(t11)
        z_raw = _val_logits(t11, "val_logits_raw.npy")[0]
        rows.append({"exp_id": "I06", "method": f"trọng số EMA ({EMA_RUN})", "model": f"{EMA_RUN} seed0", "K": 1,
                     **_metrics_row(y11, softmax_np(z_ema))})
        rows.append({"exp_id": "I06_raw", "method": f"trọng số thường cùng epoch ({EMA_RUN})", "model": f"{EMA_RUN} seed0",
                     "K": 1, **_metrics_row(y11, softmax_np(z_raw))})

    # I07 temperature scaling trên logit 1 view
    z0 = scores["I00"]
    T = inf.fit_temperature(z0, y)
    p_before, p_after = softmax_np(z0), inf.apply_temperature(z0, T)
    rows.append({"exp_id": "I07", "method": f"I00 + temperature scaling (T={T:.3f}, khớp trên val)",
                 "model": f"{rec['exp_id']} seed0", "K": 1, **_metrics_row(y, p_after),
                 "ece_before": compute_metrics(y, p_before.argmax(1), p_before)["ece"],
                 "ece_after_crossfit": _cross_fit_ece(z0, y), "T": T})
    reliability_plot(y, [p_before, p_after], ["trước TS", f"sau TS (T={T:.2f})"], figs / "reliability_val.png")

    # I08: FP16 / AMP, và gộp BN nếu kiến trúc có BN
    mean, std = _mean_std(model)
    loader = _loader(lab, val_df, f"crop{S}", mean, std)
    half = __import__("copy").deepcopy(model).half()
    names, y, outs = inf.predict_views(half, _wrap_half(loader), device)
    rows.append({"exp_id": "I08_fp16", "method": "FP16 (model.half())", "model": f"{rec['exp_id']} seed0", "K": 1,
                 **_metrics_row(y, softmax_np(outs[0]))})
    del half
    names, y, outs = inf.predict_views(model, loader, device, amp=True)
    rows.append({"exp_id": "I08_amp", "method": "AMP autocast FP16", "model": f"{rec['exp_id']} seed0", "K": 1,
                 **_metrics_row(y, softmax_np(outs[0]))})
    fusion = _fusion_rows(lab, model, cfg, val_df, device, rows)

    df = pd.DataFrame(rows)
    lat = latency_table(lab, model, cfg, fusion)
    df = _attach_latency(df, lat)
    df.to_csv(lab.root / "step3_inference.csv", index=False)
    lat.to_csv(lab.root / "step3_latency.csv", index=False)
    tradeoff_plot(df, figs / "accuracy_latency_tradeoff.png")

    # chọn cách suy luận cho chung kết: chỉ xét phương pháp một mô hình, macro-F1 val cao nhất
    single = df[df["exp_id"].isin(list(METHODS) + [f"I04_R{r}" for r in RESOLUTIONS])].dropna(subset=["val_macro_f1"])
    single = single.sort_values(["val_macro_f1", "lat_b1_p50_ms"], ascending=[False, True])
    best = single.iloc[0]
    i00 = df.set_index("exp_id").loc["I00"]
    meth = best["exp_id"].replace("I04_", "")
    lab.decide("step3_method", {
        "method": meth, "desc": best["method"], "val_macro_f1": best["val_macro_f1"],
        "delta_vs_I00": best["val_macro_f1"] - i00["val_macro_f1"], "p95_b1_ms": best["lat_b1_p95_ms"],
        "temperature_scaling": True,
        "reason": (f"macro-F1 val cao nhất trong các cách suy luận một mô hình; Δ so với I00 = "
                   f"{best['val_macro_f1'] - i00['val_macro_f1']:+.4f} (std nhiễu seed {noise:.4f}); "
                   f"p95 batch 1 = {best['lat_b1_p95_ms']:.1f} ms. Sau đó áp temperature scaling khớp trên val.")})
    return {"inference": df, "latency": lat}


class _wrap_half:
    """Loader trả ảnh FP16 (cho model.half())."""

    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for x, y, fn in self.loader:
            yield x.half(), y, fn

    def __len__(self):
        return len(self.loader)


def _fusion_rows(lab: Lab, model, cfg: Config, val_df, device, rows) -> dict:
    """Gộp BN cho model của Bước 3 nếu có BN; nếu không (ConvNeXt/ViT/Swin) thì làm trên B01 ResNet-50."""
    target, label = model, f"{cfg.exp_id} seed0 ({cfg.backbone})"
    if inf.count_bn(model) == 0:
        rows.append({"exp_id": "I08_fuse", "method": "gộp BN vào conv", "model": label, "K": 1,
                     "note": f"{cfg.backbone} không có BatchNorm (dùng LayerNorm), không gộp được"})
        target = load_trained(lab.cfg(exp_id="B01", backbone="resnet50"), device)
        label = "B01 seed0 (resnet50)"
    fused = inf.fuse_conv_bn(target.float())
    err = inf.check_fusion(target, fused, S, device=device)
    mean, std = _mean_std(target)
    loader = _loader(lab, val_df, f"crop{S}", mean, std)
    for tag, m in (("I08_unfused", target), ("I08_fused", fused)):
        names, y, outs = inf.predict_views(m, loader, device)
        rows.append({"exp_id": tag, "method": "gộp BN vào conv" if tag == "I08_fused" else "chưa gộp BN (đối chứng)",
                     "model": label, "K": 1, **_metrics_row(y, softmax_np(outs[0])),
                     "note": f"{fused.n_fused_bn} cặp conv-BN, sai số đầu ra lớn nhất {err:.2e}" if tag == "I08_fused" else ""})
    return {"model": target, "fused": fused, "label": label, "max_err": err, "n_fused": fused.n_fused_bn}


def latency_table(lab: Lab, model, cfg: Config, fusion: dict) -> pd.DataFrame:
    """Đo p50/p95/p99 (warmup 10, synchronize, 100 lần) ở batch 1 và batch 32."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = f"{cfg.exp_id} ({cfg.backbone})"
    rows = []
    for bs in (1, 32):
        for dt in ("fp32", "amp", "fp16"):
            rows.append({**latency_report(model, bs, S, dt, dev, label=f"{name} I00 {dt}"), "method": "I00" if dt == "fp32" else f"I08_{dt}"})
        rows.append({**latency_report(fusion["fused"], bs, S, "fp32", dev, fused_bn=True,
                                      label=f"{fusion['label']} gộp BN fp32"), "method": "I08_fused"})
        rows.append({**latency_report(fusion["model"], bs, S, "fp32", dev,
                                      label=f"{fusion['label']} chưa gộp BN fp32"), "method": "I08_unfused"})
        for meth in ("I01", "I02", "I02F"):
            k = METHODS[meth][1]
            for batched in (False, True):
                r = multi_forward_latency([model], bs, S, k_views=k, batched_views=batched, dtype="fp32", device=dev,
                                          label=f"{name} {meth} K={k} {'gộp batch' if batched else 'tuần tự'}")
                rows.append({**r, "method": meth if not batched else meth + "_batched"})
        if not any(t in cfg.backbone for t in TRANSFORMERS):
            for r_ in RESOLUTIONS:
                rows.append({**latency_report(model, bs, r_, "fp32", dev, label=f"{name} res {r_}"), "method": f"I04_R{r_}"})
        top3 = pd.read_csv(lab.root / "step1_backbones.csv").sort_values("val_macro_f1", ascending=False).head(3)
        models = [load_trained(lab.cfg(exp_id=e, backbone=b)) for e, b in zip(top3["exp_id"], top3["backbone"])]
        rows.append({**multi_forward_latency(models, bs, S, dtype="fp32", device=dev,
                                             label="ensemble " + "+".join(top3["backbone"])), "method": "I05a"})
        del models
        t00 = [load_trained(lab.cfg(exp_id="T00", seed=s, backbone=cfg.backbone)) for s in (0, 1, 2)]
        rows.append({**multi_forward_latency(t00, bs, S, dtype="fp32", device=dev,
                                             label="ensemble T00 x3 seed"), "method": "I05b"})
        del t00
    # các backbone của Bước 1 (đo kỹ, batch 1 và 32)
    b1 = pd.read_csv(lab.root / "step1_backbones.csv")
    for e, b in zip(b1["exp_id"], b1["backbone"]):
        m = load_trained(lab.cfg(exp_id=e, backbone=b))
        for bs in (1, 32):
            rows.append({**latency_report(m, bs, S, "fp32", dev, label=f"{e} {b}"), "method": e})
        del m
    # tính cả tiền xử lý (đọc JPEG + transform + chép lên GPU)
    _, val_df, _ = load_split(lab.labels_dir)
    mean, std = _mean_std(model)
    paths = [Path(lab.images_dir) / f for f in val_df["Filename"].head(200)]
    rows.append({**latency_with_preprocess(model, paths, build_transforms(False, S, mean=mean, std=std), dev,
                                           label=f"{name} I00 fp32 + tiền xử lý"), "method": "I00_preproc"})
    if dev == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def _attach_latency(df: pd.DataFrame, lat: pd.DataFrame) -> pd.DataFrame:
    b1 = lat[(lat["batch"] == 1) & ~lat["preprocess"]].drop_duplicates("method").set_index("method")
    b32 = lat[lat["batch"] == 32].drop_duplicates("method").set_index("method")
    alias = {"I01L": "I01", "I02L": "I02", "I07": "I00", "I08_amp": "I08_amp", "I08_fp16": "I08_fp16",
             "I06": "I00", "I06_raw": "I00"}  # EMA không đổi kiến trúc nên cùng độ trễ với 1 view
    p50, p95, p99, thr = [], [], [], []
    for e in df["exp_id"]:
        key = alias.get(e, e)
        ok = key is not None and key in b1.index
        p50.append(b1.loc[key, "p50"] if ok else np.nan)
        p95.append(b1.loc[key, "p95"] if ok else np.nan)
        p99.append(b1.loc[key, "p99"] if ok else np.nan)
        thr.append(b32.loc[key, "images_per_s"] if key is not None and key in b32.index else np.nan)
    df = df.copy()
    df["lat_b1_p50_ms"], df["lat_b1_p95_ms"], df["lat_b1_p99_ms"] = p50, p95, p99
    df["throughput_b32_img_s"] = thr
    base = df.loc[df["exp_id"] == "I00", "lat_b1_p50_ms"].iloc[0]
    df["rel_cost_vs_I00"] = df["lat_b1_p50_ms"] / base
    return df


def tradeoff_plot(df: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df.dropna(subset=["val_macro_f1", "lat_b1_p50_ms"])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(d["lat_b1_p50_ms"], d["val_macro_f1"])
    for _, r in d.iterrows():
        ax.annotate(r["exp_id"], (r["lat_b1_p50_ms"], r["val_macro_f1"]), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")
    ax.axvline(100, color="red", ls="--", lw=1, label="ngân sách 100 ms")
    ax.set_xscale("log")
    ax.set_xlabel("độ trễ p50 batch 1 (ms, log)")
    ax.set_ylabel("macro-F1 val")
    ax.set_title("Đánh đổi độ chính xác - độ trễ (Bước 3)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- #
# Bước 4
# ----------------------------------------------------------------------------- #
def step4(lab: Lab, seeds=(0, 1, 2)) -> dict:
    device = _device()
    dec = lab.load_json("decisions.json")
    meth = dec["step3_method"]["method"]
    _, val_df, test_df = load_split(lab.labels_dir)
    preload_images(pd.concat([val_df, test_df])["Filename"], lab.images_dir)
    pdir = lab.root / "predictions"
    out = []
    for seed in seeds:
        # chung kết: công thức tốt nhất + cách suy luận đã chọn + temperature scaling
        cfg = recipe_cfg(lab, exp_id="F01", seed=seed, desc="final")
        s = run(cfg)
        model = load_trained(cfg, device)
        names_v, y_v, z_v = method_scores(model, lab, val_df, meth, device)
        T = inf.fit_temperature(z_v, y_v)
        save_predictions(pdir / f"F01_seed{seed}_val.csv", names_v, y_v, inf.apply_temperature(z_v, T))
        test_file = pdir / f"F01_seed{seed}_test.csv"
        if test_file.exists():
            raise RuntimeError(f"{test_file} đã có: test chỉ chạy một lần mỗi seed")
        names_t, y_t, z_t = method_scores(model, lab, test_df, meth, device)  # test: đúng một lần
        save_predictions(pdir / f"F01_uncal_seed{seed}_test.csv", names_t, y_t, softmax_np(z_t))
        save_predictions(test_file, names_t, y_t, inf.apply_temperature(z_t, T))
        np.save(run_dir(cfg) / "test_scores.npy", z_t.astype(np.float32))
        mv = compute_metrics(y_v, z_v.argmax(1), inf.apply_temperature(z_v, T))
        out.append({"exp_id": "F01", "seed": seed, "T": T, "val_macro_f1_1view": s["val_macro_f1"],
                    "val_macro_f1": mv["macro_f1"], "val_top1": mv["top1"], "method": meth})
        del model
        # mốc: T00 + I00, dùng lại checkpoint đã train ở Bước 2, chạy test một lần
        t = run(lab.cfg(exp_id="T00", seed=seed, desc="baseline", backbone=dec["step2_recipe"]["backbone"],
                        save_test_predictions=True))
        out.append({"exp_id": "T00", "seed": seed, "T": None, "val_macro_f1": t["val_macro_f1"],
                    "val_top1": t["val_top1"], "method": "I00"})
    df = pd.DataFrame(out)
    df.to_csv(lab.root / "step4_final_val.csv", index=False)
    return {"final_val": df}


def run_eval(lab: Lab, latency_p95_ms: float | None) -> dict:
    """Gọi eval.py score / grade (nguyên bản) và lưu đầu ra vào eval_out/."""
    from train import _find_eval_dir
    ev = str(_find_eval_dir() / "eval.py")
    pdir = lab.root / "predictions"
    eout = lab.root / "eval_out"
    test_csv = str(Path(lab.labels_dir) / "test_subset0.csv")
    val_csv = str(Path(lab.labels_dir) / "val_subset0.csv")
    labels = str(Path(lab.labels_dir) / "labels.csv")
    logs = {}
    for tag, pat in (("F01", "F01_seed*_test.csv"), ("T00", "T00_seed*_test.csv"),
                     ("F01_uncal", "F01_uncal_seed*_test.csv")):
        cmd = [sys.executable, ev, "score", "--pred", str(pdir / pat), "--test-csv", test_csv,
               "--labels", labels, "--tag", tag, "--out", str(eout)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        logs[f"score_{tag}"] = r.stdout + r.stderr
        print(r.stdout, r.stderr)
    cmd = [sys.executable, ev, "grade", "--final", str(pdir / "F01_seed*_test.csv"),
           "--baseline", str(pdir / "T00_seed*_test.csv"), "--uncal", str(pdir / "F01_uncal_seed*_test.csv"),
           "--final-val", str(pdir / "F01_seed*_val.csv"), "--val-csv", val_csv,
           "--test-csv", test_csv, "--labels", labels, "--out", str(eout)]
    if latency_p95_ms is not None:
        cmd += ["--latency-p95-ms", f"{latency_p95_ms:.2f}", "--latency-method", "proper"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    logs["grade"] = r.stdout + r.stderr
    print(r.stdout, r.stderr)
    eout.mkdir(parents=True, exist_ok=True)
    for k, v in logs.items():
        (eout / f"{k}.txt").write_text(v, encoding="utf-8")
    return logs


def confusion_and_errors(lab: Lab) -> None:
    """Ma trận nhầm lẫn trên test (cộng 3 seed) cho F01 và T00, và ảnh bị đoán sai của hai lớp khó."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    eout, figs = lab.root / "eval_out", lab.root / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    short = ["Chinee", "Lantana", "Parkins.", "Parthen.", "P.acacia", "R.vine", "Siam", "Snake", "Negative"]
    for tag in ("F01", "T00"):
        cm = pd.read_csv(eout / f"{tag}_confusion_sum.csv", index_col=0).to_numpy()
        norm = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(9):
            for j in range(9):
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.1%}" if cm[i, j] else "", ha="center", va="center",
                        fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(9), short, rotation=45, ha="right")
        ax.set_yticks(range(9), short)
        ax.set_xlabel("dự đoán")
        ax.set_ylabel("nhãn thật")
        ax.set_title(f"Ma trận nhầm lẫn test, {tag} (cộng 3 seed; % theo hàng)")
        fig.tight_layout()
        fig.savefig(figs / f"confusion_test_{tag}.png", dpi=120)
        plt.close(fig)

    p = pd.read_csv(lab.root / "predictions" / "F01_seed0_test.csv")
    wrong = p[(p["y_true"] != p["y_pred"]) & p["y_true"].isin(HARD)]
    wrong = wrong.assign(conf=p.loc[wrong.index, [f"p{i}" for i in range(9)]].max(1)).sort_values("conf", ascending=False)
    pick = pd.concat([wrong[wrong["y_true"] == c].head(6) for c in HARD])
    if len(pick):
        fig, axes = plt.subplots(2, 6, figsize=(15, 6))
        for ax in axes.flat:
            ax.axis("off")
        for r, c in enumerate(HARD):
            sub = pick[pick["y_true"] == c]
            for j, (_, row) in enumerate(sub.iterrows()):
                with Image.open(Path(lab.images_dir) / row["Filename"]) as im:
                    axes[r, j].imshow(im.convert("RGB"))
                axes[r, j].set_title(f"thật: {CLASS_NAMES[c]}\nđoán: {CLASS_NAMES[int(row['y_pred'])]} "
                                     f"({row['conf']:.2f})", fontsize=8)
        fig.suptitle("Ảnh test bị đoán sai (F01 seed 0), độ tin cậy cao nhất trước")
        fig.tight_layout()
        fig.savefig(figs / "errors_hard_classes.png", dpi=100)
        plt.close(fig)
