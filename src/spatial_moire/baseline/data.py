from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
POSITIVES_DIR = "moire"
NEGATIVES_DIR = "gt"
MOIRE_LABEL = 1
CLEAN_LABEL = 0


def list_image_files(directory: Path) -> list[Path]:
    files = [path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda path: path.name)


def list_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    moire_files = {path.name: path for path in list_image_files(split_dir / POSITIVES_DIR)}
    gt_files = {path.name: path for path in list_image_files(split_dir / NEGATIVES_DIR)}
    if set(moire_files) != set(gt_files):
        raise ValueError(f"moire/gt filename mismatch under {split_dir}")
    return [(moire_files[name], gt_files[name]) for name in sorted(moire_files)]


def samples_from_pairs(pairs: list[tuple[Path, Path]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for moire_path, gt_path in pairs:
        samples.append({"path": str(moire_path), "label": MOIRE_LABEL})
        samples.append({"path": str(gt_path), "label": CLEAN_LABEL})
    return samples


def split_train_val(
    train_dir: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = list_pairs(split_dir=train_dir)
    random.Random(seed).shuffle(pairs)
    val_count = round(len(pairs) * val_fraction)
    train_samples = samples_from_pairs(pairs=pairs[val_count:])
    val_samples = samples_from_pairs(pairs=pairs[:val_count])
    return train_samples, val_samples


def collect_test_samples(test_dir: Path) -> list[dict[str, Any]]:
    return samples_from_pairs(pairs=list_pairs(split_dir=test_dir))


def save_split_manifest(
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    path: Path,
    seed: int = 42,
    val_fraction: float = 0.2,
) -> None:
    manifest = {
        "seed": seed,
        "val_fraction": val_fraction,
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "train": train_samples,
        "val": val_samples,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data=json.dumps(manifest, indent=2), encoding="utf-8")


class moire_dataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], transform: transforms.Compose | None = None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        sample = self.samples[index]
        image = Image.open(sample["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(sample["label"])


def build_train_transform(crop_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=crop_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(resize_size: int = 256, crop_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(size=(resize_size, resize_size)),
            transforms.CenterCrop(size=crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
