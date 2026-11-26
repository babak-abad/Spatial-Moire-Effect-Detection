"""Sanity checks for the on-the-fly moire augmentation label semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spatial_moire.baseline.augmentation import augmented_moire_dataset  # noqa: E402
from spatial_moire.baseline.data import build_train_transform  # noqa: E402

TRAIN_DIR = REPO_ROOT / "datasets" / "LCDMoire" / "my_dataset_final" / "train"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
NUM_SAMPLES = 6


def load_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def build_samples(kind: str, label: int) -> list[dict]:
    directory = TRAIN_DIR / kind
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)[:NUM_SAMPLES]
    if not paths:
        raise SystemExit(f"no images found in {directory}")
    return [{"path": str(path), "label": label} for path in paths]


def check_augmented_clean_images_become_moire() -> None:
    samples = build_samples(kind="gt", label=0)
    dataset = augmented_moire_dataset(samples=samples, transform=None, augment_prob=1.0, seed=42)
    for index in range(len(dataset)):
        image, label = dataset[index]
        assert label == 1, f"augmented clean image {index} must carry the moire label"
        assert not np.array_equal(to_array(image), load_array(Path(samples[index]["path"]))), (
            f"clean image {index} was not modified by augmentation"
        )
    print(f"ok: {len(dataset)} clean images augmented on the fly, all relabeled as moire")


def check_zero_probability_keeps_labels_and_pixels() -> None:
    samples = build_samples(kind="gt", label=0)
    dataset = augmented_moire_dataset(samples=samples, transform=None, augment_prob=0.0, seed=42)
    for index in range(len(dataset)):
        image, label = dataset[index]
        assert label == 0, f"clean image {index} must keep the clean label when augmentation is disabled"
        assert np.array_equal(to_array(image), load_array(Path(samples[index]["path"]))), (
            f"clean image {index} must be unmodified when augmentation is disabled"
        )
    print("ok: augment_prob=0 keeps labels and pixels untouched")


def check_real_moire_images_pass_through() -> None:
    samples = build_samples(kind="moire", label=1)
    dataset = augmented_moire_dataset(samples=samples, transform=None, augment_prob=1.0, seed=42)
    for index in range(len(dataset)):
        image, label = dataset[index]
        assert label == 1, f"real moire image {index} must keep the moire label"
        assert np.array_equal(to_array(image), load_array(Path(samples[index]["path"]))), (
            f"real moire image {index} must never be augmented"
        )
    print("ok: real moire images pass through unaugmented with label 1")


def check_same_seed_is_reproducible() -> None:
    samples = build_samples(kind="gt", label=0)
    dataset_a = augmented_moire_dataset(samples=samples, transform=None, augment_prob=1.0, seed=123)
    dataset_b = augmented_moire_dataset(samples=samples, transform=None, augment_prob=1.0, seed=123)
    for index in range(len(dataset_a)):
        image_a, label_a = dataset_a[index]
        image_b, label_b = dataset_b[index]
        assert (label_a, label_b) == (1, 1), f"image {index} must be augmented and labeled moire in both runs"
        assert np.array_equal(to_array(image_a), to_array(image_b)), f"image {index} differs across same-seed runs"
    print("ok: identical seeds produce identical augmentations")


def check_transform_pipeline_output() -> None:
    samples = build_samples(kind="gt", label=0)
    dataset = augmented_moire_dataset(
        samples=samples,
        transform=build_train_transform(),
        augment_prob=1.0,
        seed=7,
    )
    image, label = dataset[0]
    assert label == 1, "augmented clean image must carry the moire label after the train transform"
    assert isinstance(image, torch.Tensor), "train transform must return a tensor"
    assert tuple(image.shape) == (3, 224, 224), f"unexpected tensor shape {tuple(image.shape)}"
    print("ok: augmented images flow through the train transform as (3, 224, 224) tensors")


def main() -> None:
    check_augmented_clean_images_become_moire()
    check_zero_probability_keeps_labels_and_pixels()
    check_real_moire_images_pass_through()
    check_same_seed_is_reproducible()
    check_transform_pipeline_output()
    print("all augmentation sanity checks passed")


if __name__ == "__main__":
    main()
