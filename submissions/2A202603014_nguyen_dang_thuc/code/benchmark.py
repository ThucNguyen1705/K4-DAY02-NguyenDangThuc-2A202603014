"""benchmark.py - đo độ trễ suy luận (slide trang 73, 75; GUIDE.md mục 4.1).

Quy tắc em dùng:
  - warmup 10 lần đầu, bỏ kết quả
  - torch.cuda.synchronize() ngay TRƯỚC và ngay SAU đoạn đo
  - >= 50 lần đo (mặc định 100), báo cáo p50 / p95 / p99
  - mặc định KHÔNG tính tiền xử lý: đầu vào là tensor đã nằm trên GPU. Riêng hàm
    latency_with_preprocess đo cả đọc ảnh + transform + chép lên GPU để so sánh.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 60, sync=None) -> dict:
    """Đo `fn()` (mili-giây). `sync` = torch.cuda.synchronize trên GPU, None trên CPU."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "p99": float(np.percentile(t, 99)), "mean": float(t.mean()), "n": iters}


def _prepare(model, dtype: str, device: str):
    m = copy.deepcopy(model).to(device).eval()
    if dtype == "fp16":
        m = m.half()
    return m


def _forward_fn(model, x, dtype: str):
    def fn():
        with torch.inference_mode():
            if dtype == "amp":
                with torch.autocast(device_type=x.device.type, dtype=torch.float16):
                    model(x)
            else:
                model(x)
    return fn


def _meta(device: str) -> dict:
    gpu = torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else "cpu"
    return {"gpu": gpu, "torch": torch.__version__}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 60, fused_bn: bool = False, label: str = "") -> dict:
    """Độ trễ forward với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    dtype: "fp32" | "amp" (autocast FP16) | "fp16" (model.half()).
    """
    m = _prepare(model, dtype, device)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    if dtype == "fp16":
        x = x.half()
    if device == "cuda":
        m = m.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last)
    sync = torch.cuda.synchronize if device == "cuda" else None
    r = bench(_forward_fn(m, x, dtype), warmup, iters, sync)
    del m
    return {"config": label, **_meta(device), "dtype": dtype, "batch": batch_size, "img_size": img_size,
            "fused_bn": fused_bn, "preprocess": False, "p50": r["p50"], "p95": r["p95"], "p99": r["p99"],
            "mean": r["mean"], "n": r["n"], "images_per_s": batch_size / (r["p50"] / 1000.0)}


def multi_forward_latency(models, batch_size: int, img_size: int, k_views: int = 1, batched_views: bool = False,
                          view_size: int | None = None, dtype: str = "fp32", device: str = "cuda",
                          warmup: int = 10, iters: int = 60, label: str = "") -> dict:
    """Độ trễ của TTA / ensemble: chạy lần lượt từng model, mỗi model K view.

    batched_views=False: K lượt forward riêng (giống cách làm đơn giản nhất);
    batched_views=True : gộp K view thành một batch K*batch_size, một lượt forward.
    """
    ms = [_prepare(m, dtype, device) for m in models]
    s = view_size or img_size
    x = torch.randn(batch_size, 3, s, s, device=device)
    if dtype == "fp16":
        x = x.half()
    if device == "cuda":
        ms = [m.to(memory_format=torch.channels_last) for m in ms]
    sync = torch.cuda.synchronize if device == "cuda" else None

    def fn():
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16,
                                                     enabled=dtype == "amp"):
            for m in ms:
                if batched_views:
                    m(x.repeat(k_views, 1, 1, 1))
                else:
                    for _ in range(k_views):
                        m(x)

    r = bench(fn, warmup, iters, sync)
    del ms
    return {"config": label, **_meta(device), "dtype": dtype, "batch": batch_size, "img_size": s,
            "fused_bn": False, "preprocess": False, "k": k_views * len(models),
            "p50": r["p50"], "p95": r["p95"], "p99": r["p99"], "mean": r["mean"], "n": r["n"],
            "images_per_s": batch_size / (r["p50"] / 1000.0)}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ TTA K view (K lượt forward riêng), kèm tỉ số so với K * p50 của 1 view."""
    single = latency_report(model, kw.get("batch_size", 1), kw.get("img_size", 224), kw.get("dtype", "fp32"),
                            kw.get("device", "cuda"))
    tta = multi_forward_latency([model], kw.get("batch_size", 1), kw.get("img_size", 224), k_views=k_views,
                                dtype=kw.get("dtype", "fp32"), device=kw.get("device", "cuda"))
    tta["ratio_vs_k_single"] = tta["p50"] / (k_views * single["p50"])
    return tta


def latency_with_preprocess(model, image_paths, transform, device: str = "cuda", warmup: int = 10,
                            iters: int = 60, label: str = "") -> dict:
    """Batch 1, tính cả đọc JPEG + transform (CPU) + chép lên GPU + forward FP32."""
    from PIL import Image
    m = _prepare(model, "fp32", device)
    if device == "cuda":
        m = m.to(memory_format=torch.channels_last)
    paths = list(image_paths)
    it = [0]
    sync = torch.cuda.synchronize if device == "cuda" else None

    def fn():
        p = paths[it[0] % len(paths)]
        it[0] += 1
        with Image.open(p) as im:
            x = transform(im.convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            m(x)

    r = bench(fn, warmup, iters, sync)
    return {"config": label, **_meta(device), "dtype": "fp32", "batch": 1, "img_size": None,
            "fused_bn": False, "preprocess": True, "p50": r["p50"], "p95": r["p95"], "p99": r["p99"],
            "mean": r["mean"], "n": r["n"], "images_per_s": 1 / (r["p50"] / 1000.0)}
