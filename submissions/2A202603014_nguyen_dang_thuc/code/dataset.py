"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Giao diện (giữ đúng như starter/):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)

Thêm so với khung: preload_images (giải mã ảnh một lần, giữ trong RAM để các lần chạy sau
không phải đọc JPEG lại), denormalize (để vẽ ảnh sau augmentation).
"""
from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TOTAL_IMAGES = 17509
EXPECTED_FRAC = {"train": 0.6, "val": 0.2, "test": 0.2}
# Table 1 của bài báo, dùng để đối chiếu ở bước EDA
PAPER_COUNTS = [1125, 1064, 1031, 1022, 1062, 1009, 1074, 1016, 9106]
# Val/test: resize về img_size / CROP_PCT rồi center crop img_size. Với 224 thì resize 256 = kích thước gốc.
CROP_PCT = 0.875


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train/val/test_subset{fold}.csv nguyên bản (S1), không sửa, không lọc."""
    labels_dir = Path(labels_dir)
    out = []
    for part in ("train", "val", "test"):
        df = pd.read_csv(labels_dir / f"{part}_subset{fold}.csv")
        missing = {"Filename", "Label"} - set(df.columns)
        if missing:
            raise ValueError(f"{part}_subset{fold}.csv thiếu cột {missing}")
        out.append(df)
    return tuple(out)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, expected_total: int = TOTAL_IMAGES, tol: float = 0.01,
                verbose: bool = True) -> dict:
    """Các kiểm tra bắt buộc trước khi train (README mục 2.1). Sai thì dừng ngay bằng AssertionError."""
    parts = {"train": train_df, "val": val_df, "test": test_df}
    n = {k: len(v) for k, v in parts.items()}
    total = sum(n.values())
    frac = {k: v / total for k, v in n.items()}

    for k, v in parts.items():
        dup = int(v["Filename"].duplicated().sum())
        assert dup == 0, f"{k}: có {dup} Filename bị trùng trong cùng một tập"
        bad = set(v["Label"].unique()) - set(range(NUM_CLASSES))
        assert not bad, f"{k}: nhãn ngoài 0..8: {bad}"

    per_class = pd.DataFrame({k: v["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
                              for k, v in parts.items()})
    per_class.index = CLASS_NAMES
    per_class["total"] = per_class.sum(1)
    per_class["paper_table1"] = PAPER_COUNTS

    sets = {k: set(v["Filename"]) for k, v in parts.items()}
    overlap = {"train∩val": len(sets["train"] & sets["val"]),
               "train∩test": len(sets["train"] & sets["test"]),
               "val∩test": len(sets["val"] & sets["test"])}
    union = sets["train"] | sets["val"] | sets["test"]

    on_disk = set(os.listdir(images_dir))
    missing = sorted(union - on_disk)

    warnings = [f"{k}: {frac[k]:.4f} lệch > {tol:.0%} so với {EXPECTED_FRAC[k]}"
                for k in parts if abs(frac[k] - EXPECTED_FRAC[k]) > tol]

    if verbose:
        print("Số ảnh mỗi tập:", n, "| tỉ lệ:", {k: round(v, 4) for k, v in frac.items()})
        print("Giao từng cặp tập:", overlap)
        print(f"Hợp ba tập: {len(union)} ảnh (kỳ vọng {expected_total}); thiếu trên đĩa: {len(missing)}")
        print(per_class.to_string())
        for w in warnings:
            print("CẢNH BÁO:", w)

    assert all(v == 0 for v in overlap.values()), f"giao giữa các tập khác rỗng: {overlap}"
    assert len(union) == expected_total, f"hợp ba tập = {len(union)}, kỳ vọng {expected_total}"
    assert not missing, f"{len(missing)} file trong CSV không có trong {images_dir}, ví dụ {missing[:5]}"

    return {"n": n, "frac": frac, "per_class": per_class, "overlap": overlap,
            "union": len(union), "missing": len(missing), "warnings": warnings}


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD, eval_mode: str = "crop"):
    """Tạo transform.

    train=True, `aug`:
      - "basic"  : RandomResizedCrop(img_size) + lật ngang
      - "color"  : basic + ColorJitter(0.3, 0.3, 0.3, 0.05)
      - "trivial": basic + TrivialAugmentWide
      - "randaug": basic + RandAugment(2, 9)
      - "vflip"  : basic + lật dọc
    train=False:
      - eval_mode="crop": Resize(img_size / 0.875) + CenterCrop(img_size). Ở 224 là ảnh gốc 256 cắt giữa 224.
      - eval_mode="full": Resize(img_size), giữ toàn bộ ảnh (dùng cho TTA nhiều crop).
    """
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    if not train:
        if eval_mode == "full":
            return T.Compose([T.Resize((img_size, img_size)), *norm])
        resize = int(round(img_size / CROP_PCT))
        return T.Compose([T.Resize(resize), T.CenterCrop(img_size), *norm])

    base = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
    extra = {
        "basic": [],
        "color": [T.ColorJitter(0.3, 0.3, 0.3, 0.05)],
        "trivial": [T.TrivialAugmentWide()],
        "randaug": [T.RandAugment(num_ops=2, magnitude=9)],
        "vflip": [T.RandomVerticalFlip()],
    }
    if aug not in extra:
        raise ValueError(f"aug không hỗ trợ: {aug}; chọn trong {list(extra)}")
    return T.Compose([*base, *extra[aug], *norm])


def denormalize(x: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """(C, H, W) hoặc (N, C, H, W) đã chuẩn hoá -> ảnh trong [0, 1] để vẽ."""
    m = torch.tensor(mean, device=x.device).view(-1, 1, 1)
    s = torch.tensor(std, device=x.device).view(-1, 1, 1)
    return (x * s + m).clamp(0, 1)


# ----------------------------------------------------------------------------- #
# Cache ảnh trong RAM: 17.509 ảnh 256x256x3 uint8 khoảng 3,4 GB.
# Giải mã một lần cho cả phiên notebook; DataLoader worker (fork) đọc chung bộ nhớ.
# ----------------------------------------------------------------------------- #
_CACHE: dict[str, np.ndarray] = {}


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def preload_images(filenames, images_dir: str | Path, threads: int = 8) -> dict:
    """Nạp trước ảnh vào cache. Trả về thống kê kích thước/kênh (dùng cho EDA)."""
    images_dir = Path(images_dir)
    todo = [f for f in dict.fromkeys(filenames) if f not in _CACHE]
    with ThreadPoolExecutor(threads) as ex:
        for f, arr in zip(todo, ex.map(lambda f: _load_rgb(images_dir / f), todo)):
            _CACHE[f] = arr
    shapes = pd.Series([_CACHE[f].shape for f in filenames]).value_counts()
    return {str(k): int(v) for k, v in shapes.items()}


def image_modes(filenames, images_dir: str | Path) -> dict:
    """Chế độ màu gốc của ảnh (RGB, L, ...) trước khi convert."""
    images_dir = Path(images_dir)
    modes = {}
    for f in filenames:
        with Image.open(images_dir / f) as im:
            modes[im.mode] = modes.get(im.mode, 0) + 1
    return modes


class DeepWeedsDataset(Dataset):
    """Đọc ảnh theo DataFrame (Filename, Label). __getitem__ -> (tensor, int(label), filename)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.filenames = df["Filename"].tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        name = self.filenames[i]
        arr = _CACHE.get(name)
        img = Image.fromarray(arr) if arr is not None else Image.open(self.images_dir / name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], name


def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2, seed: int = 0):
    """DataLoader. train=False thì không xáo, giữ đúng thứ tự df để ghép logit với Filename."""
    ds = DeepWeedsDataset(df, images_dir, transform)
    g = torch.Generator()
    g.manual_seed(seed)
    smp, shuffle = None, train
    if train and sampler == "balanced":
        counts = np.bincount(ds.labels, minlength=NUM_CLASSES)
        w = 1.0 / counts[np.asarray(ds.labels)]
        smp = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(ds),
                                    replacement=True, generator=g)
        shuffle = False
    elif sampler not in (None, "balanced"):
        raise ValueError(f"sampler không hỗ trợ: {sampler}")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=smp,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=train, worker_init_fn=_seed_worker, generator=g,
                      persistent_workers=False)
