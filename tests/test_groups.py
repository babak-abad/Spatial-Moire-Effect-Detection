"""Sanity checks for the grouped capture-style augmentation pipeline."""

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

from config import (  # noqa: E402
    DEFAULT_DATA_DIR,
    GROUP_AUGMENT_COUNT_RANGE,
    IMAGE_EXTENSIONS,
    TRAIN_CROP_SIZE,
    TRAIN_SPLIT_DIR,
)
from spatial_moire.baseline.augmentation import augmented_moire_dataset  # noqa: E402
from spatial_moire.baseline.data import build_train_transform  # noqa: E402
from spatial_moire.moire_types import generator_names_for_type  # noqa: E402
from spatial_moire.synthesis.augment import MOIRE_GENERATORS  # noqa: E402
from spatial_moire.synthesis.groups import (  # noqa: E402
    AUGMENTATION_GROUPS,
    apply_group_augmentation,
    normalized_group_count_range,
    selected_group_names,
)

TRAIN_DIR = DEFAULT_DATA_DIR / TRAIN_SPLIT_DIR
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 96
NUM_SAMPLES = 6
DRAWS = 200


def build_synthetic_image() -> np.ndarray:
    grids = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH].astype(dtype=np.float32)
    red = 0.2 + 0.6 * (grids[1] / IMAGE_WIDTH)
    green = 0.2 + 0.6 * (grids[0] / IMAGE_HEIGHT)
    blue = 0.8 - 0.5 * red
    image = np.stack(arrays=[red, green, blue], axis=2)
    return np.clip(a=image, a_min=0.0, a_max=1.0).astype(dtype=np.float32)


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


def check_registry_holds_grouped_factories() -> None:
    assert normalized_group_count_range(num_groups_range=GROUP_AUGMENT_COUNT_RANGE) == GROUP_AUGMENT_COUNT_RANGE
    assert len(AUGMENTATION_GROUPS) >= 5, "the group registry must stay populated with several groups"
    for group_name, factories in AUGMENTATION_GROUPS.items():
        assert factories, f"group {group_name} must hold at least one factory"
        for factory in factories:
            augmenter = factory(seed=1)
            assert augmenter.name, f"{group_name} factory must build a named imgaug augmenter"
    assert set(selected_group_names()) == set(AUGMENTATION_GROUPS), "default selection must cover every group"
    print(f"ok: registry holds {len(AUGMENTATION_GROUPS)} groups with seed-built imgaug augmenters")


def check_every_group_factory_preserves_shape() -> None:
    image = build_synthetic_image()
    for group_name, factories in AUGMENTATION_GROUPS.items():
        for factory in factories:
            augmenter = factory(seed=11)
            raw = np.asarray(augmenter.augment_image(image=image))
            assert raw.shape == image.shape, f"{group_name} must preserve the image shape"
            assert raw.dtype == image.dtype, f"{group_name} must keep float32 dtype"
            assert np.isfinite(raw).all(), f"{group_name} must not create non-finite values"
    print("ok: every group factory preserves shape, dtype and finiteness on float32 input")


def check_pipeline_stays_in_unit_range() -> None:
    image = build_synthetic_image()
    rng = np.random.default_rng(seed=3)
    for _ in range(DRAWS):
        result, params = apply_group_augmentation(image=image, rng=rng, num_groups_range=(3, 3))
        assert result.shape == image.shape and result.dtype == np.float32, "pipeline output must be HxWx3 float32"
        assert result.min() >= 0.0 and result.max() <= 1.0, "pipeline output must stay in [0, 1]"
        drawn = [entry["group"] for entry in params["groups"]]
        assert params["num_groups"] == 3 and len(drawn) == 3, "a fixed (3, 3) count must apply exactly three groups"
        assert len(set(drawn)) == 3, "groups must be drawn without replacement"
    print("ok: pipeline output stays [0, 1] float32 with three unique groups per call")


def check_subset_and_count_are_honored() -> None:
    image = build_synthetic_image()
    subset = ["noise", "blur", "contrast"]
    rng = np.random.default_rng(seed=5)
    sequences: list[tuple[str, ...]] = []
    for _ in range(DRAWS):
        _result, params = apply_group_augmentation(
            image=image,
            rng=rng,
            group_subset=subset,
            num_groups_range=(2, 2),
        )
        drawn = tuple(entry["group"] for entry in params["groups"])
        assert params["num_groups"] == 2 and len(drawn) == 2, "a fixed (2, 2) count must apply exactly two groups"
        assert set(drawn) <= set(subset), "drawn groups must stay inside the requested subset"
        sequences.append(drawn)
    assert len(set(sequences)) > 1, "group choice and order must vary across draws"
    print("ok: group subsets and fixed counts are honored with varying orders")


def check_zero_groups_is_identity() -> None:
    image = build_synthetic_image()
    result, params = apply_group_augmentation(image=image, rng=np.random.default_rng(seed=6), num_groups_range=(0, 0))
    assert params["num_groups"] == 0 and params["groups"] == []
    assert np.array_equal(result, image), "a zero group count must leave the image bit-identical"
    print("ok: a zero group count is an exact identity")


def check_same_seed_is_reproducible() -> None:
    image = build_synthetic_image()
    result_a, params_a = apply_group_augmentation(image=image, rng=np.random.default_rng(seed=123))
    result_b, params_b = apply_group_augmentation(image=image, rng=np.random.default_rng(seed=123))
    result_c, params_c = apply_group_augmentation(image=image, rng=np.random.default_rng(seed=124))
    assert np.array_equal(result_a, result_b), "identical seeds must produce identical images"
    assert params_a == params_b, "identical seeds must produce identical recipes"
    assert params_a != params_c, "different seeds must draw different recipes"
    print("ok: identical seeds reproduce images and recipes, different seeds differ")


def check_bad_inputs_are_rejected() -> None:
    image = build_synthetic_image()
    attempts = (
        lambda: apply_group_augmentation(image=image.astype(np.float64), rng=np.random.default_rng(seed=0)),
        lambda: apply_group_augmentation(image=image[:, :, :1], rng=np.random.default_rng(seed=0)),
        lambda: apply_group_augmentation(image=image, rng=np.random.default_rng(seed=0), num_groups_range=(2, 1)),
        lambda: apply_group_augmentation(image=image, rng=np.random.default_rng(seed=0), num_groups_range=(-1, 2)),
        lambda: apply_group_augmentation(image=image, rng=np.random.default_rng(seed=0), group_subset=["rainbow"]),
        lambda: selected_group_names(group_subset=[]),
        lambda: normalized_group_count_range(num_groups_range=(3, 1)),
    )
    for attempt in attempts:
        try:
            attempt()
        except ValueError:
            continue
        raise AssertionError("invalid group pipeline inputs must raise ValueError")
    print("ok: invalid images, count ranges and group names raise ValueError")


def check_groups_compose_after_moire_synthesis() -> None:
    image = build_synthetic_image()
    rng = np.random.default_rng(seed=7)
    generator_name = str(rng.choice(a=generator_names_for_type(type_name="pixel_grid")))
    moire_image, _params = MOIRE_GENERATORS[generator_name](image=image, rng=rng)
    grouped, group_params = apply_group_augmentation(image=moire_image, rng=rng)
    assert grouped.shape == image.shape and grouped.dtype == np.float32
    assert grouped.min() >= 0.0 and grouped.max() <= 1.0
    assert group_params["num_groups"] >= 1, "the default count range must apply at least one group"
    print(f"ok: group augmentation composes cleanly on moire synthesis output ({generator_name})")


def check_augmented_dataset_applies_groups() -> None:
    samples = build_samples(kind="gt", label=0)
    plain = augmented_moire_dataset(samples=samples, transform=None, augment_prob=1.0, seed=42)
    grouped = augmented_moire_dataset(samples=samples, transform=None, augment_prob=1.0, seed=42, group_augment=True)
    repeat = augmented_moire_dataset(samples=samples, transform=None, augment_prob=1.0, seed=42, group_augment=True)
    for index in range(len(grouped)):
        plain_image, plain_label = plain[index]
        grouped_image, grouped_label = grouped[index]
        repeat_image, _repeat_label = repeat[index]
        assert (plain_label, grouped_label) == (1, 1), f"image {index} must be relabeled as moire in both modes"
        assert np.array_equal(to_array(grouped_image), to_array(repeat_image)), f"image {index} must be reproducible"
        assert not np.array_equal(to_array(grouped_image), to_array(plain_image)), (
            f"image {index}: group augmentation must change pixels relative to plain moire augmentation"
        )
        assert not np.array_equal(to_array(grouped_image), load_array(Path(samples[index]["path"]))), (
            f"image {index} must differ from the clean source"
        )
    print(f"ok: {len(grouped)} clean images get reproducible group-augmented moire")


def check_real_moire_images_pass_through_with_groups() -> None:
    samples = build_samples(kind="moire", label=1)
    dataset = augmented_moire_dataset(samples=samples, transform=None, augment_prob=1.0, group_augment=True, seed=42)
    for index in range(len(dataset)):
        image, label = dataset[index]
        assert label == 1, f"real moire image {index} must keep the moire label"
        assert np.array_equal(to_array(image), load_array(Path(samples[index]["path"]))), (
            f"real moire image {index} must never be group-augmented"
        )
    print("ok: real moire images pass through unaugmented even with groups enabled")


def check_transform_pipeline_output_with_groups() -> None:
    samples = build_samples(kind="gt", label=0)
    dataset = augmented_moire_dataset(
        samples=samples,
        transform=build_train_transform(),
        augment_prob=1.0,
        seed=7,
        group_augment=True,
    )
    image, label = dataset[0]
    assert label == 1, "group-augmented clean image must carry the moire label after the train transform"
    assert isinstance(image, torch.Tensor), "train transform must return a tensor"
    assert tuple(image.shape) == (3, TRAIN_CROP_SIZE, TRAIN_CROP_SIZE), f"unexpected tensor shape {tuple(image.shape)}"
    print(f"ok: group-augmented images flow through the train transform as (3, {TRAIN_CROP_SIZE}, {TRAIN_CROP_SIZE}) tensors")


def check_dataset_rejects_invalid_group_settings() -> None:
    samples = build_samples(kind="gt", label=0)
    attempts = (
        lambda: augmented_moire_dataset(samples=samples, transform=None, augment_prob=1.0, group_names=["rainbow"]),
        lambda: augmented_moire_dataset(samples=samples, transform=None, augment_prob=1.0, num_groups_range=(3, 1)),
    )
    for attempt in attempts:
        try:
            attempt()
        except ValueError:
            continue
        raise AssertionError("invalid dataset group settings must raise ValueError")
    print("ok: dataset rejects unknown group names and invalid count ranges")


def main() -> None:
    check_registry_holds_grouped_factories()
    check_every_group_factory_preserves_shape()
    check_pipeline_stays_in_unit_range()
    check_subset_and_count_are_honored()
    check_zero_groups_is_identity()
    check_same_seed_is_reproducible()
    check_bad_inputs_are_rejected()
    check_groups_compose_after_moire_synthesis()
    check_augmented_dataset_applies_groups()
    check_real_moire_images_pass_through_with_groups()
    check_transform_pipeline_output_with_groups()
    check_dataset_rejects_invalid_group_settings()
    print("all group augmentation checks passed")


if __name__ == "__main__":
    main()