from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import (
    CLEAN_IMAGE_FOLDERS,
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
    MOIRE_IMAGE_FOLDERS,
    MOIRE_LABEL,
    RANDOM_RESIZED_CROP_SCALE,
    TEST_FRACTION,
    TRAIN_CROP_SIZE,
    TRAIN_FRACTION,
    VAL_FRACTION,
)

# Class-label constants are re-exported from config for backward compatibility
# with existing imports of this module.


def list_image_files(directory: Path) -> list[Path]:
    files = [path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda path: path.name)


def list_folder_pairs(
    clean_folders: list[Path] | None = None,
    moire_folders: list[Path] | None = None,
) -> list[list[tuple[Path, Path]]]:
    """Filename-matched (moire, clean) pairs, one list per index-aligned folder pair.

    clean_folders[i] and moire_folders[i] must hold the same scenes matched by
    filename (LCDMoire gt / moire twins), so a scene pair is the indivisible
    split unit that keeps both twins inside one split. Defaults to the config
    pools. Raises ValueError on unequal list lengths, missing or empty folders
    or a moire/clean filename mismatch.
    """
    clean_pool = CLEAN_IMAGE_FOLDERS if clean_folders is None else clean_folders
    moire_pool = MOIRE_IMAGE_FOLDERS if moire_folders is None else moire_folders
    if len(clean_pool) != len(moire_pool):
        raise ValueError(
            f"clean/moire folder lists must be index-aligned: {len(clean_pool)} clean vs {len(moire_pool)} moire folders"
        )
    pair_groups: list[list[tuple[Path, Path]]] = []
    for clean_dir, moire_dir in zip(clean_pool, moire_pool, strict=True):
        for directory in (clean_dir, moire_dir):
            if not directory.is_dir():
                raise ValueError(f"configured image folder does not exist: {directory}")
        moire_files = {path.name: path for path in list_image_files(directory=moire_dir)}
        clean_files = {path.name: path for path in list_image_files(directory=clean_dir)}
        if not moire_files:
            raise ValueError(f"configured moire image folder holds no supported images: {moire_dir}")
        if not clean_files:
            raise ValueError(f"configured clean image folder holds no supported images: {clean_dir}")
        if set(moire_files) != set(clean_files):
            raise ValueError(f"moire/clean filename mismatch between {moire_dir} and {clean_dir}")
        pair_groups.append([(moire_files[name], clean_files[name]) for name in sorted(moire_files)])
    return pair_groups


def samples_from_pairs(pairs: list[tuple[Path, Path]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for moire_path, gt_path in pairs:
        samples.append({"path": str(moire_path), "label": MOIRE_LABEL})
        samples.append({"path": str(gt_path), "label": CLEAN_LABEL})
    return samples


def validate_split_fractions(train_fraction: float, val_fraction: float, test_fraction: float) -> None:
    """Raise ValueError unless all three split fractions lie in [0, 1] and sum to 1.0."""
    fractions = (train_fraction, val_fraction, test_fraction)
    if any(fraction < 0.0 or fraction > 1.0 for fraction in fractions):
        raise ValueError(f"split fractions must lie in [0, 1]: {fractions}")
    if not math.isclose(sum(fractions), 1.0):
        raise ValueError(f"split fractions must sum to 1.0: {fractions}")


def split_train_val_test(
    clean_folders: list[Path] | None = None,
    moire_folders: list[Path] | None = None,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the pooled scene pairs into train / val / test sample lists.

    All folder pairs are pooled (folder-pair order, then filename order) and
    shuffled at pair level with random.Random(seed), so the moire and clean
    twins of one scene always land in the same split. Train and val pair counts
    are rounded from the fractions; test receives the remainder, so every
    pooled pair ends up in exactly one split.
    """
    validate_split_fractions(train_fraction=train_fraction, val_fraction=val_fraction, test_fraction=test_fraction)
    pair_groups = list_folder_pairs(clean_folders=clean_folders, moire_folders=moire_folders)
    pairs = [pair for pair_group in pair_groups for pair in pair_group]
    random.Random(seed).shuffle(pairs)
    train_count = round(len(pairs) * train_fraction)
    val_count = round(len(pairs) * val_fraction)
    train_samples = samples_from_pairs(pairs=pairs[:train_count])
    val_samples = samples_from_pairs(pairs=pairs[train_count : train_count + val_count])
    test_samples = samples_from_pairs(pairs=pairs[train_count + val_count :])
    return train_samples, val_samples, test_samples


def save_split_manifest(
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    test_samples: list[dict[str, Any]],
    path: Path,
    seed: int = DEFAULT_SEED,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> None:
    """Persist the exact split (seed, fractions and all three sample lists) as JSON."""
    manifest = {
        "seed": seed,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data=json.dumps(manifest, indent=JSON_INDENT), encoding=JSON_ENCODING)


def load_split_manifest(path: Path) -> dict[str, Any] | None:
    """Read a saved split manifest, or None when path does not exist."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding=JSON_ENCODING))


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
