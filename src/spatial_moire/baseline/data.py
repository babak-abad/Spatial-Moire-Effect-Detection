from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from config import (
    CLEAN_IMAGE_FOLDERS_TEST,
    CLEAN_IMAGE_FOLDERS_TRAIN,
    CLEAN_IMAGE_FOLDERS_VAL,
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
    MOIRE_IMAGE_FOLDERS_TEST,
    MOIRE_IMAGE_FOLDERS_TRAIN,
    MOIRE_IMAGE_FOLDERS_VAL,
    MOIRE_LABEL,
    RANDOM_RESIZED_CROP_SCALE,
    TRAIN_CROP_SIZE,
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


def split_datasets(
    clean_train_folders: list[Path] | None = None,
    moire_train_folders: list[Path] | None = None,
    clean_val_folders: list[Path] | None = None,
    moire_val_folders: list[Path] | None = None,
    clean_test_folders: list[Path] | None = None,
    moire_test_folders: list[Path] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect the train / val / test splits from the six per-class folder lists.

    Each argument defaults to its config counterpart. A split's sample list is
    the clean list's samples followed by the moire list's samples, both in
    (folder, filename) order. Validation: every list must be non-empty; no
    folder may appear in two of a class's three lists, so a source folder
    never feeds two splits; collect_class_samples raises for missing or empty
    folders. There is no pairing between the classes and a sample's label
    comes from its list alone.
    """
    clean_train = CLEAN_IMAGE_FOLDERS_TRAIN if clean_train_folders is None else clean_train_folders
    moire_train = MOIRE_IMAGE_FOLDERS_TRAIN if moire_train_folders is None else moire_train_folders
    clean_val = CLEAN_IMAGE_FOLDERS_VAL if clean_val_folders is None else clean_val_folders
    moire_val = MOIRE_IMAGE_FOLDERS_VAL if moire_val_folders is None else moire_val_folders
    clean_test = CLEAN_IMAGE_FOLDERS_TEST if clean_test_folders is None else clean_test_folders
    moire_test = MOIRE_IMAGE_FOLDERS_TEST if moire_test_folders is None else moire_test_folders
    for class_name, lists in (
        ("clean", (clean_train, clean_val, clean_test)),
        ("moire", (moire_train, moire_val, moire_test)),
    ):
        for split_name, folders in zip(("train", "val", "test"), lists, strict=True):
            if not folders:
                raise ValueError(f"the {class_name} {split_name} image folder list must not be empty")
        seen: set[Path] = set()
        for folders in lists:
            for folder in folders:
                if folder in seen:
                    raise ValueError(
                        f"{class_name} image folder {folder} is listed for more than one split; "
                        "a source folder may feed at most one split"
                    )
                seen.add(folder)
    train_samples = collect_class_samples(folders=clean_train, label=CLEAN_LABEL)
    train_samples += collect_class_samples(folders=moire_train, label=MOIRE_LABEL)
    val_samples = collect_class_samples(folders=clean_val, label=CLEAN_LABEL)
    val_samples += collect_class_samples(folders=moire_val, label=MOIRE_LABEL)
    test_samples = collect_class_samples(folders=clean_test, label=CLEAN_LABEL)
    test_samples += collect_class_samples(folders=moire_test, label=MOIRE_LABEL)
    return train_samples, val_samples, test_samples


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
