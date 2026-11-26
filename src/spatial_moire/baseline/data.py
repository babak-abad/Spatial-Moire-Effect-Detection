from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import (
    CLEAN_LABEL,
    COLOR_JITTER_BRIGHTNESS,
    COLOR_JITTER_CONTRAST,
    COLOR_JITTER_HUE,
    COLOR_JITTER_SATURATION,
    DEFAULT_SEED,
    EVAL_CROP_SIZE,
    EVAL_RESIZE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_COLOR_MODE,
    IMAGE_EXTENSIONS,
    JSON_ENCODING,
    JSON_INDENT,
    MOIRE_LABEL,
    NEGATIVES_DIR,
    POSITIVES_DIR,
    RANDOM_RESIZED_CROP_SCALE,
    TRAIN_CROP_SIZE,
    VAL_FRACTION,
)

# Dataset-layout and label constants are re-exported from config for backward
# compatibility with existing imports of this module.


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
    val_fraction: float = VAL_FRACTION,
    seed: int = DEFAULT_SEED,
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
    seed: int = DEFAULT_SEED,
    val_fraction: float = VAL_FRACTION,
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
    path.write_text(data=json.dumps(manifest, indent=JSON_INDENT), encoding=JSON_ENCODING)


class moire_dataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], transform: transforms.Compose | None = None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        sample = self.samples[index]
        image = Image.open(sample["path"]).convert(mode=IMAGE_COLOR_MODE)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(sample["label"])


def build_train_transform(crop_size: int = TRAIN_CROP_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=crop_size, scale=RANDOM_RESIZED_CROP_SCALE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=COLOR_JITTER_BRIGHTNESS,
                contrast=COLOR_JITTER_CONTRAST,
                saturation=COLOR_JITTER_SATURATION,
                hue=COLOR_JITTER_HUE,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(
    resize_size: int = EVAL_RESIZE_SIZE,
    crop_size: int = EVAL_CROP_SIZE,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(size=(resize_size, resize_size)),
            transforms.CenterCrop(size=crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
