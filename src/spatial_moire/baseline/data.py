from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler
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
    FOLDER_UNIT_MAX_IMAGES,
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


def collect_class_samples(folders: list[Path], label: int) -> list[dict[str, Any]]:
    """Walk one per-class folder pool and emit {"path", "label"} dicts.

    Every folder must exist and hold at least one supported image — the loud
    ValueErrors stay. Files are listed in filename order via list_image_files
    and the label comes from the pool alone, never from image content or
    augmentation.
    """
    samples: list[dict[str, Any]] = []
    for directory in folders:
        if not directory.is_dir():
            raise ValueError(f"configured image folder does not exist: {directory}")
        files = list_image_files(directory=directory)
        if not files:
            raise ValueError(f"configured image folder holds no supported images: {directory}")
        samples.extend({"path": str(path), "label": label} for path in files)
    return samples


def build_split_units(folders: list[Path], label: int) -> list[list[dict[str, Any]]]:
    """Split units of one class stream, each unit a list of {"path", "label"} dicts.

    A folder holding at most FOLDER_UNIT_MAX_IMAGES images is one atomic unit.
    A larger folder contributes one unit per image, and images sharing a
    filename stem across the larger folders of one stream merge into a single
    unit, so captures of the same scene cannot straddle splits. Units keep
    pool order (folder order, then filename order).
    """
    samples = collect_class_samples(folders=folders, label=label)
    by_folder: dict[Path, list[dict[str, Any]]] = {}
    for sample in samples:
        by_folder.setdefault(Path(sample["path"]).parent, []).append(sample)
    units: list[list[dict[str, Any]]] = []
    stem_units: dict[str, list[dict[str, Any]]] = {}
    for folder in folders:
        folder_samples = by_folder.get(folder, [])
        if not folder_samples:
            continue
        if len(folder_samples) <= FOLDER_UNIT_MAX_IMAGES:
            units.append(folder_samples)
            continue
        for sample in folder_samples:
            stem = Path(sample["path"]).stem
            unit = stem_units.get(stem)
            if unit is None:
                unit = []
                stem_units[stem] = unit
                units.append(unit)
            unit.append(sample)
    return units


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
    """Split each class stream's units independently into train / val / test.

    The clean and moire pools are collected and grouped into split units
    separately (collect_class_samples / build_split_units), then shuffled with
    two independent shuffles — random.Random(seed) for the clean stream,
    random.Random(seed + 1) for the moire stream. Each stream is cut by the
    fractions: train and val unit counts are rounded, test receives the
    remainder. Every split's sample list is its clean units' samples followed
    by its moire units' samples; there is no pairing between the streams and
    a sample's label comes from its pool alone.
    """
    validate_split_fractions(train_fraction=train_fraction, val_fraction=val_fraction, test_fraction=test_fraction)
    clean_pool = CLEAN_IMAGE_FOLDERS if clean_folders is None else clean_folders
    moire_pool = MOIRE_IMAGE_FOLDERS if moire_folders is None else moire_folders
    clean_units = build_split_units(folders=clean_pool, label=CLEAN_LABEL)
    moire_units = build_split_units(folders=moire_pool, label=MOIRE_LABEL)
    random.Random(seed).shuffle(clean_units)
    random.Random(seed + 1).shuffle(moire_units)
    train_samples: list[dict[str, Any]] = []
    val_samples: list[dict[str, Any]] = []
    test_samples: list[dict[str, Any]] = []
    for units in (clean_units, moire_units):
        train_count = round(len(units) * train_fraction)
        val_count = round(len(units) * val_fraction)
        for split_samples, unit_slice in (
            (train_samples, units[:train_count]),
            (val_samples, units[train_count : train_count + val_count]),
            (test_samples, units[train_count + val_count :]),
        ):
            split_samples.extend(sample for unit in unit_slice for sample in unit)
    return train_samples, val_samples, test_samples


SPLIT_MANIFEST_ALGORITHM: str = "per_class_units_v1"


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
    """Persist the exact split (algorithm, seed, fractions and all three sample lists) as JSON."""
    manifest = {
        "algorithm": SPLIT_MANIFEST_ALGORITHM,
        "seed": seed,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "folder_unit_max_images": FOLDER_UNIT_MAX_IMAGES,
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


class balanced_moire_dataset(moire_dataset):
    """moire_dataset whose virtual index space interleaves clean and moire slots.

    Even indices address the clean pool and odd indices the moire pool, both
    cycling modulo their pool size, so every contiguous even-length window of
    range(len(dataset)) — exactly what a DataLoader forms from sequential or
    balanced_shuffle_sampler order — holds half clean and half moire items,
    however unbalanced the source samples are. The smaller pool is oversampled
    (it repeats within one epoch); __len__ is 2 * max(n_clean, n_moire).
    Raises ValueError when either class pool is empty.
    """

    def __init__(self, samples: list[dict[str, Any]], transform: transforms.Compose | None = None) -> None:
        super().__init__(samples=samples, transform=transform)
        self.clean_pool = [sample for sample in samples if int(sample["label"]) == CLEAN_LABEL]
        self.moire_pool = [sample for sample in samples if int(sample["label"]) == MOIRE_LABEL]
        if not self.clean_pool or not self.moire_pool:
            raise ValueError(
                "balanced_moire_dataset requires at least one clean and one moire sample, "
                f"got {len(self.clean_pool)} clean / {len(self.moire_pool)} moire"
            )
        self.num_slots = max(len(self.clean_pool), len(self.moire_pool))

    def __len__(self) -> int:
        return 2 * self.num_slots

    def __getitem__(self, index: int) -> tuple[Any, int]:
        slot, want_moire = divmod(index, 2)
        pool = self.moire_pool if want_moire else self.clean_pool
        sample = pool[slot % len(pool)]
        image = Image.open(sample["path"]).convert(mode=IMAGE_COLOR_MODE)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(sample["label"])


class balanced_shuffle_sampler(Sampler[int]):
    """Shuffles balanced_moire_dataset slots while keeping (clean, moire) pairs adjacent.

    __iter__ yields the dataset's interleaved index space (2k, 2k + 1) over a
    seeded permutation of the slot order, so every epoch visits the slots in a
    fresh random order while every batch of even size drawn from the produced
    sequence stays exactly half clean and half moire. The first iteration uses
    seed, each following iteration (epoch) uses seed + epoch, so training stays
    reproducible yet reshuffles per epoch.
    """

    def __init__(self, num_slots: int, seed: int = DEFAULT_SEED) -> None:
        if num_slots < 1:
            raise ValueError(f"num_slots must be at least 1, got {num_slots}")
        self.num_slots = num_slots
        self.seed = seed
        self._epoch = 0

    def __iter__(self) -> Iterator[int]:
        order = np.random.default_rng(seed=self.seed + self._epoch).permutation(self.num_slots)
        self._epoch += 1
        for slot in order:
            yield 2 * int(slot)
            yield 2 * int(slot) + 1

    def __len__(self) -> int:
        return 2 * self.num_slots


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
