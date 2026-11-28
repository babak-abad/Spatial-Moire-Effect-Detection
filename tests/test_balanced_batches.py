"""Sanity checks for balanced training batches (half clean / half moire drawn from the real class pools)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import CLEAN_IMAGE_FOLDERS, IMAGE_EXTENSIONS, MOIRE_IMAGE_FOLDERS  # noqa: E402
from spatial_moire.baseline.augmentation import augmented_moire_dataset  # noqa: E402
from spatial_moire.baseline.data import (  # noqa: E402
    balanced_moire_dataset,
    balanced_shuffle_sampler,
    moire_dataset,
)

SAMPLE_DIRS: dict[str, Path] = {"gt": CLEAN_IMAGE_FOLDERS[0], "moire": MOIRE_IMAGE_FOLDERS[0]}
BATCH_SIZE = 16
TO_TENSOR = transforms.Compose([transforms.ToTensor()])


def load_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def image_paths(kind: str, count: int) -> list[Path]:
    directory = SAMPLE_DIRS[kind]
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)[:count]
    if len(paths) < count:
        raise SystemExit(f"need at least {count} images in {directory}, found {len(paths)}")
    return paths


def class_samples(clean_count: int, moire_count: int) -> list[dict]:
    samples = [{"path": str(path), "label": 0} for path in image_paths(kind="gt", count=clean_count)]
    samples += [{"path": str(path), "label": 1} for path in image_paths(kind="moire", count=moire_count)]
    return samples


def balanced_loader(
    dataset: balanced_moire_dataset | augmented_moire_dataset,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=balanced_shuffle_sampler(num_slots=dataset.num_slots, seed=42),
        num_workers=num_workers,
    )


def check_parity_mapping_balanced_dataset() -> None:
    samples = class_samples(clean_count=3, moire_count=11)
    dataset = balanced_moire_dataset(samples=samples, transform=None)
    assert dataset.num_slots == 11, "num_slots must be the larger pool size"
    assert len(dataset) == 22, "length must be 2 * max(n_clean, n_moire)"
    for index in range(len(dataset)):
        _image, label = dataset[index]
        expected_label = index % 2
        assert label == expected_label, f"index {index} (parity {expected_label}) returned label {label}"
    print("ok: even indices map to clean, odd to moire; length is 2 * max(pool sizes)")


def check_exact_split_through_real_dataloader() -> None:
    samples = class_samples(clean_count=12, moire_count=12)
    dataset = balanced_moire_dataset(samples=samples, transform=TO_TENSOR)
    loader = balanced_loader(dataset=dataset)
    batches = [(len(labels), int(labels.sum())) for _images, labels in loader]
    assert batches == [(16, 8), (8, 4)], f"expected (size, moire) pairs (16, 8) and (8, 4), got {batches}"
    print("ok: a real DataLoader yields 16-item batches with exactly 8 clean + 8 moire (trailing 8-item batch: 4 + 4)")


def check_unbalanced_sources_stay_balanced() -> None:
    samples = class_samples(clean_count=3, moire_count=11)
    dataset = balanced_moire_dataset(samples=samples, transform=TO_TENSOR)
    for batch_index, (_images, labels) in enumerate(balanced_loader(dataset=dataset)):
        moire_count = int(labels.sum())
        assert 2 * moire_count == len(labels), (
            f"batch {batch_index} from 3 clean / 11 moire sources holds {moire_count} moire of {len(labels)}"
        )
    print("ok: unbalanced sources (3 clean vs 11 moire) still yield exactly half/half batches")


def check_balanced_augmentation_semantics() -> None:
    samples = class_samples(clean_count=4, moire_count=4)
    clean_paths = [sample["path"] for sample in samples if sample["label"] == 0]
    moire_paths = [sample["path"] for sample in samples if sample["label"] == 1]
    always = augmented_moire_dataset(samples=samples, transform=None, moire_augment_prob=1.0, seed=3, balanced=True)
    never = augmented_moire_dataset(samples=samples, transform=None, moire_augment_prob=0.0, seed=3, balanced=True)
    assert len(always) == len(never) == 8, "balanced augmentation must keep the 2 * max(pool) index space"
    for index in range(8):
        slot = index // 2
        if index % 2:
            always_image, always_label = always[index]
            never_image, never_label = never[index]
            assert (always_label, never_label) == (1, 1), f"moire slot {index} must carry the moire label"
            assert np.array_equal(to_array(never_image), load_array(Path(moire_paths[slot]))), (
                f"moire slot {index} with moire_augment_prob=0 must pass the real moire source through unchanged"
            )
            assert not np.array_equal(to_array(always_image), load_array(Path(moire_paths[slot]))), (
                f"moire slot {index} with moire_augment_prob=1 must gain a pattern on top of the real moire source"
            )
        else:
            always_image, always_label = always[index]
            never_image, never_label = never[index]
            assert (always_label, never_label) == (0, 0), f"clean slot {index} must keep the clean label"
            assert np.array_equal(to_array(never_image), load_array(Path(clean_paths[slot]))), (
                f"clean slot {index} must pass through unchanged"
            )
            assert np.array_equal(to_array(always_image), to_array(never_image)), (
                f"clean slot {index} must never receive moire synthesis, whatever moire_augment_prob is"
            )
    print("ok: moire_augment_prob=1 patterns every moire slot, prob=0 passes real moire through, clean slots stay clean")


def check_balanced_augmented_batches() -> None:
    samples = class_samples(clean_count=10, moire_count=10)
    dataset = augmented_moire_dataset(samples=samples, transform=TO_TENSOR, moire_augment_prob=0.5, seed=7, balanced=True)
    assert len(dataset) == 20, "balanced augmented dataset must expose 2 * num_slots items"
    for batch_index, (_images, labels) in enumerate(balanced_loader(dataset=dataset)):
        moire_count = int(labels.sum())
        assert 2 * moire_count == len(labels), f"augmented batch {batch_index} holds {moire_count} moire of {len(labels)}"
    for index in range(0, len(dataset), 2):
        _image, label = dataset[index]
        assert label == 0, f"clean slot {index} must never carry the moire label"
    print("ok: augmented balanced dataset yields exactly half/half batches and never labels clean slots as moire")


def check_balanced_batches_with_workers() -> None:
    samples = class_samples(clean_count=12, moire_count=12)
    dataset = augmented_moire_dataset(samples=samples, transform=TO_TENSOR, moire_augment_prob=0.5, seed=42, balanced=True)
    for batch_index, (_images, labels) in enumerate(balanced_loader(dataset=dataset, num_workers=2)):
        moire_count = int(labels.sum())
        assert 2 * moire_count == len(labels), f"worker batch {batch_index} holds {moire_count} moire of {len(labels)}"
    print("ok: num_workers=2 still yields exactly half/half balanced batches")


def check_sampler_properties() -> None:
    sampler = balanced_shuffle_sampler(num_slots=12, seed=42)
    order = list(iter(sampler))
    assert len(order) == 24 and sorted(order) == list(range(24)), "sampler must cover every index exactly once"
    for position in range(0, 24, 2):
        assert order[position] // 2 == order[position + 1] // 2, f"pair {position // 2} is not adjacent"
    for start in range(24 - 6 + 1):
        window = order[start : start + 6]
        assert sum(index % 2 for index in window) == 3, f"6-window at offset {start} is not half/half"
    assert list(iter(balanced_shuffle_sampler(num_slots=12, seed=42))) == order, "same seed must reproduce the order"
    next_epoch = list(iter(sampler))
    assert next_epoch != order and sorted(next_epoch) == list(range(24)), "the next epoch must reshuffle without loss"
    assert list(iter(balanced_shuffle_sampler(num_slots=12, seed=43))) != order, "different seeds must draw different orders"
    print("ok: sampler covers all indices once, keeps pairs adjacent, balances every window, reshuffles per epoch")


def check_invalid_pools_are_rejected() -> None:
    attempts = (
        lambda: balanced_moire_dataset(samples=class_samples(clean_count=0, moire_count=4), transform=None),
        lambda: balanced_moire_dataset(samples=class_samples(clean_count=4, moire_count=0), transform=None),
        lambda: augmented_moire_dataset(
            samples=class_samples(clean_count=4, moire_count=0), transform=None, moire_augment_prob=0.0, balanced=True
        ),
        lambda: augmented_moire_dataset(
            samples=class_samples(clean_count=4, moire_count=0), transform=None, moire_augment_prob=1.0, balanced=True
        ),
        lambda: balanced_shuffle_sampler(num_slots=0, seed=42),
    )
    for attempt in attempts:
        try:
            attempt()
        except ValueError:
            continue
        raise AssertionError("empty class pools and zero slots must raise ValueError")
    print("ok: empty class pools raise ValueError; balanced mode never falls back to clean-source synthesis")


def check_default_mode_is_unchanged() -> None:
    samples = class_samples(clean_count=4, moire_count=4)
    plain = moire_dataset(samples=samples, transform=None)
    augmented = augmented_moire_dataset(samples=samples, transform=None, moire_augment_prob=0.0, seed=42)
    assert len(plain) == len(augmented) == 8, "balanced=False must keep the plain index space"
    for index in range(8):
        _plain_image, plain_label = plain[index]
        _augmented_image, augmented_label = augmented[index]
        assert plain_label == augmented_label == samples[index]["label"], (
            f"index {index} must keep the source label order when balanced=False"
        )
    print("ok: balanced=False keeps the legacy length and source label order")


def main() -> None:
    check_parity_mapping_balanced_dataset()
    check_exact_split_through_real_dataloader()
    check_unbalanced_sources_stay_balanced()
    check_balanced_augmentation_semantics()
    check_balanced_augmented_batches()
    check_balanced_batches_with_workers()
    check_sampler_properties()
    check_invalid_pools_are_rejected()
    check_default_mode_is_unchanged()
    print("all balanced batch sanity checks passed")


if __name__ == "__main__":
    main()
