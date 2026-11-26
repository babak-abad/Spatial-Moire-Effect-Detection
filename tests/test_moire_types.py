"""Sanity checks for the weighted moire type registry and weighted sampling."""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spatial_moire.baseline.augmentation import augmented_moire_dataset  # noqa: E402
from spatial_moire.moire_types import (  # noqa: E402
    MOIRE_TYPES,
    generator_names_for_type,
    sample_moire_type,
    total_weight,
    type_probabilities,
)
from spatial_moire.synthesis.augment import MOIRE_GENERATORS  # noqa: E402

EXPECTED_WEIGHTS = {
    "pixel_grid": 24.0,
    "color": 20.0,
    "scanline": 6.0,
    "temporal": 8.0,
    "perspective": 12.0,
    "rotational": 12.0,
    "defocus": 8.0,
    "demosaic": 6.0,
    "nested": 4.0,
}
NEW_GENERATORS = ("scanline_bands", "temporal_bands", "defocus_moire", "nested_moire")
DRAWS = 200_000
TOLERANCE = 0.005
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 96


def build_synthetic_image() -> np.ndarray:
    grids = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH].astype(dtype=np.float32)
    red = 0.2 + 0.6 * (grids[1] / IMAGE_WIDTH)
    green = 0.2 + 0.6 * (grids[0] / IMAGE_HEIGHT)
    blue = 0.8 - 0.5 * red
    image = np.stack(arrays=[red, green, blue], axis=2)
    return np.clip(a=image, a_min=0.0, a_max=1.0).astype(dtype=np.float32)


def check_registry_matches_documented_types() -> None:
    assert set(MOIRE_TYPES) == set(EXPECTED_WEIGHTS), "registry must hold exactly the nine documented types"
    assert total_weight() == 100.0, f"documented weights must total 100, got {total_weight()}"
    for name, weight in EXPECTED_WEIGHTS.items():
        info = MOIRE_TYPES[name]
        assert info.weight == weight, f"{name} weight must stay {weight}, got {info.weight}"
        assert info.title and info.cause and info.appearance, f"{name} must carry documentation fields"
        assert info.generator_names, f"{name} must map to at least one generator"
        for generator_name in info.generator_names:
            assert generator_name in MOIRE_GENERATORS, f"{name} references unknown generator {generator_name}"
    for generator_name in NEW_GENERATORS:
        assert generator_name in MOIRE_GENERATORS, f"missing new generator {generator_name}"
    print("ok: registry holds the nine documented types, weights total 100, all generators resolve")


def check_possibility_is_weight_over_total() -> None:
    probabilities = type_probabilities()
    total = total_weight()
    assert math.isclose(sum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-9), "possibilities must sum to 1"
    for name, info in MOIRE_TYPES.items():
        expected = info.weight / total
        assert math.isclose(probabilities[name], expected, rel_tol=0.0, abs_tol=1e-12), (
            f"{name} possibility must equal weight / total_weight ({expected}), got {probabilities[name]}"
        )
    print("ok: every possibility equals weight / total_weight and the values sum to 1.0")


def check_subset_renormalizes() -> None:
    subset = ["pixel_grid", "color", "nested"]
    probabilities = type_probabilities(type_names=subset)
    assert math.isclose(sum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(probabilities["pixel_grid"], 24.0 / 48.0, rel_tol=0.0, abs_tol=1e-12)
    rng = np.random.default_rng(seed=5)
    draws = {sample_moire_type(rng=rng, type_names=subset) for _ in range(500)}
    assert draws <= set(subset), "subset sampling must never return an unselected type"
    print("ok: type subsets renormalize weights over the selection only")


def check_weighted_sampling_matches_possibilities() -> None:
    rng = np.random.default_rng(seed=42)
    draws = [sample_moire_type(rng=rng) for _ in range(DRAWS)]
    probabilities = type_probabilities()
    for name, possibility in probabilities.items():
        empirical = draws.count(name) / DRAWS
        assert abs(empirical - possibility) <= TOLERANCE, (
            f"{name} empirical share {empirical:.4f} drifted from possibility {possibility:.4f}"
        )
    print("ok: weighted sampling frequencies match the possibilities within tolerance")


def check_unknown_type_names_are_rejected() -> None:
    attempts = (
        lambda: type_probabilities(type_names=["pixel_grid", "rainbow"]),
        lambda: sample_moire_type(rng=np.random.default_rng(seed=0), type_names=["rainbow"]),
        lambda: generator_names_for_type(type_name="rainbow"),
    )
    for attempt in attempts:
        try:
            attempt()
        except ValueError:
            continue
        raise AssertionError("unknown moire type names must raise ValueError")
    print("ok: unknown type names raise ValueError across the registry API")


def check_every_type_synthesizes_valid_moire() -> None:
    image = build_synthetic_image()
    rng = np.random.default_rng(seed=7)
    for name in MOIRE_TYPES:
        for generator_name in generator_names_for_type(type_name=name):
            augmented, params = MOIRE_GENERATORS[generator_name](image=image.copy(), rng=rng)
            assert augmented.shape == image.shape, f"{generator_name} must preserve the image shape"
            assert augmented.dtype == np.float32, f"{generator_name} must return float32"
            assert augmented.min() >= 0.0 and augmented.max() <= 1.0, f"{generator_name} must stay in [0, 1]"
            assert params, f"{generator_name} must return non-empty params"
    print("ok: every type's generators produce valid [0, 1] float32 images with params")


def check_sampling_is_deterministic_per_seed() -> None:
    sequence_a = [sample_moire_type(rng=np.random.default_rng(seed=123)) for _ in range(1000)]
    sequence_b = [sample_moire_type(rng=np.random.default_rng(seed=123)) for _ in range(1000)]
    assert sequence_a == sequence_b, "identical seeds must produce identical type sequences"
    print("ok: identical seeds produce identical weighted type sequences")


def check_augmented_dataset_draws_weighted_types() -> None:
    source = (np.clip(a=build_synthetic_image(), a_min=0.0, a_max=1.0) * 255.0).round().astype(dtype=np.uint8)
    with tempfile.TemporaryDirectory() as temp_dir:
        sample_path = Path(temp_dir) / "sample.png"
        Image.fromarray(obj=source).save(fp=sample_path)
        samples = [{"path": str(sample_path), "label": 0}]
        dataset = augmented_moire_dataset(samples=samples, transform=None, augment_prob=1.0, seed=42)
        augmented_image, label = dataset[0]
        assert label == 1, "augmented clean image must carry the moire label"
        assert not np.array_equal(np.asarray(augmented_image, dtype=np.uint8), source), (
            "weighted type augmentation must modify the pixels"
        )
        restricted = augmented_moire_dataset(
            samples=samples,
            transform=None,
            augment_prob=1.0,
            type_names=["scanline"],
            seed=42,
        )
        restricted_image, restricted_label = restricted[0]
        assert restricted_label == 1, "type-restricted augmentation must still label the image as moire"
        assert not np.array_equal(np.asarray(restricted_image, dtype=np.uint8), source), (
            "type-restricted augmentation must still modify the pixels"
        )
        try:
            augmented_moire_dataset(samples=samples, transform=None, augment_prob=1.0, type_names=["rainbow"])
        except ValueError:
            print("ok: dataset draws weighted moire types, honors subsets and rejects unknown names")
        else:
            raise AssertionError("augmented_moire_dataset must reject unknown type names")


def main() -> None:
    check_registry_matches_documented_types()
    check_possibility_is_weight_over_total()
    check_subset_renormalizes()
    check_weighted_sampling_matches_possibilities()
    check_unknown_type_names_are_rejected()
    check_every_type_synthesizes_valid_moire()
    check_sampling_is_deterministic_per_seed()
    check_augmented_dataset_draws_weighted_types()
    print("all moire type registry checks passed")


if __name__ == "__main__":
    main()