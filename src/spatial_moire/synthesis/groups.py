"""Group-based capture-style augmentation applied after moire synthesis.

Inspired by the grouped imgaug recipes of image_text_reader/engine/augment.py:
related pixel-level augmenters live in named groups (noise, photometric,
contrast, blur, dropouts, geometric), one augmenter is drawn per selected
group, and a random subset of the groups is applied in random order after
moire synthesis. Unlike the engine, the subset draw, the per-group choice and
every child augmenter seed come from the caller's numpy Generator, so
identical seeds keep producing identical images, and the applied recipe is
returned as a JSON-friendly params dict for synthesis manifests.

Every augmenter was probed on float32 [0, 1] images (imgaug targets uint8):
most results stay in range, but a few (Multiply, additive noise, blur edges)
overflow, so each step is clipped back to [0, 1] float32. imgaug's Salt /
Pepper family is excluded because it writes raw 0-255 values on float arrays
(pepper turns into white specks after clipping).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from config import (
    GROUP_AUGMENT_COUNT_RANGE,
    GROUP_BLUR_AVERAGE_KERNEL_RANGE,
    GROUP_BLUR_GAUSSIAN_SIGMA_RANGE,
    GROUP_BLUR_MOTION_ANGLE_RANGE,
    GROUP_BLUR_MOTION_KERNEL,
    GROUP_CONTRAST_GAMMA_RANGE,
    GROUP_CONTRAST_LINEAR_ALPHA_RANGE,
    GROUP_CONTRAST_SIGMOID_CUTOFF_RANGE,
    GROUP_CONTRAST_SIGMOID_GAIN_RANGE,
    GROUP_DROPOUT_COARSE_PROBABILITY_RANGE,
    GROUP_DROPOUT_COARSE_SIZE_PERCENT_RANGE,
    GROUP_DROPOUT_PIXEL_PROBABILITY_RANGE,
    GROUP_GEOMETRIC_FILL_VALUE,
    GROUP_GEOMETRIC_ROTATION_RANGE,
    GROUP_GEOMETRIC_SCALE_RANGE,
    GROUP_GEOMETRIC_SHEAR_RANGE,
    GROUP_GEOMETRIC_TRANSLATE_PERCENT_RANGE,
    GROUP_NOISE_GAUSSIAN_SIGMA_RANGE,
    GROUP_NOISE_LAPLACE_SCALE_RANGE,
    GROUP_PHOTOMETRIC_ADD_RANGE,
    GROUP_PHOTOMETRIC_MULTIPLY_RANGE,
)

# numpy 2.x removed np.sctypes / np.bool / np.complex; imgaug 0.4.0 reads them
# at import time (imgaug.py:45-47), so patch them before importing imgaug.
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "bool": [np.bool_],
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
    }
if not hasattr(np, "bool"):
    np.bool = np.bool_  # noqa: A001
if not hasattr(np, "complex"):
    np.complex = np.complex128  # noqa: A001

from imgaug import augmenters as iaa  # noqa: E402
from imgaug.augmenters.meta import Augmenter  # noqa: E402

augmenter_factory = Callable[[int], Augmenter]

AUGMENTATION_GROUPS: dict[str, tuple[augmenter_factory, ...]] = {
    "noise": (
        lambda seed: iaa.AdditiveGaussianNoise(scale=GROUP_NOISE_GAUSSIAN_SIGMA_RANGE, seed=seed),
        lambda seed: iaa.AdditiveGaussianNoise(scale=GROUP_NOISE_GAUSSIAN_SIGMA_RANGE, per_channel=True, seed=seed),
        lambda seed: iaa.AdditiveLaplaceNoise(scale=GROUP_NOISE_LAPLACE_SCALE_RANGE, seed=seed),
    ),
    "photometric": (
        lambda seed: iaa.Multiply(mul=GROUP_PHOTOMETRIC_MULTIPLY_RANGE, seed=seed),
        lambda seed: iaa.Multiply(mul=GROUP_PHOTOMETRIC_MULTIPLY_RANGE, per_channel=0.5, seed=seed),
        lambda seed: iaa.AddElementwise(value=GROUP_PHOTOMETRIC_ADD_RANGE, seed=seed),
    ),
    "contrast": (
        lambda seed: iaa.GammaContrast(gamma=GROUP_CONTRAST_GAMMA_RANGE, seed=seed),
        lambda seed: iaa.LinearContrast(alpha=GROUP_CONTRAST_LINEAR_ALPHA_RANGE, seed=seed),
        lambda seed: iaa.SigmoidContrast(
            gain=GROUP_CONTRAST_SIGMOID_GAIN_RANGE,
            cutoff=GROUP_CONTRAST_SIGMOID_CUTOFF_RANGE,
            seed=seed,
        ),
    ),
    "blur": (
        lambda seed: iaa.GaussianBlur(sigma=GROUP_BLUR_GAUSSIAN_SIGMA_RANGE, seed=seed),
        lambda seed: iaa.AverageBlur(k=GROUP_BLUR_AVERAGE_KERNEL_RANGE, seed=seed),
        lambda seed: iaa.MotionBlur(k=GROUP_BLUR_MOTION_KERNEL, angle=GROUP_BLUR_MOTION_ANGLE_RANGE, seed=seed),
    ),
    "dropouts": (
        lambda seed: iaa.CoarseDropout(
            p=GROUP_DROPOUT_COARSE_PROBABILITY_RANGE,
            size_percent=GROUP_DROPOUT_COARSE_SIZE_PERCENT_RANGE,
            seed=seed,
        ),
        lambda seed: iaa.Dropout(p=GROUP_DROPOUT_PIXEL_PROBABILITY_RANGE, per_channel=0.5, seed=seed),
    ),
    "geometric": (
        lambda seed: iaa.Affine(
            scale=GROUP_GEOMETRIC_SCALE_RANGE,
            rotate=GROUP_GEOMETRIC_ROTATION_RANGE,
            order=1,
            mode="constant",
            cval=GROUP_GEOMETRIC_FILL_VALUE,
            backend="skimage",
            seed=seed,
        ),
        lambda seed: iaa.Affine(
            shear=GROUP_GEOMETRIC_SHEAR_RANGE,
            translate_percent=GROUP_GEOMETRIC_TRANSLATE_PERCENT_RANGE,
            order=1,
            mode="constant",
            cval=GROUP_GEOMETRIC_FILL_VALUE,
            backend="skimage",
            seed=seed,
        ),
    ),
}


def selected_group_names(group_subset: list[str] | None = None) -> list[str]:
    """Resolve and validate the active augmentation group names (all groups by default)."""
    selected = list(AUGMENTATION_GROUPS) if group_subset is None else list(group_subset)
    unknown_names = [name for name in selected if name not in AUGMENTATION_GROUPS]
    if unknown_names:
        raise ValueError(f"unknown augmentation group names: {unknown_names}; expected one of {sorted(AUGMENTATION_GROUPS)}")
    if not selected:
        raise ValueError("at least one augmentation group must be selected")
    return selected


def normalized_group_count_range(num_groups_range: tuple[int, int]) -> tuple[int, int]:
    """Validate an inclusive (low, high) group-count range and return it unchanged."""
    low, high = num_groups_range
    if low < 0 or high < low:
        raise ValueError(f"num_groups_range must be (low, high) with 0 <= low <= high, got {num_groups_range}")
    return (low, high)


def apply_group_augmentation(
    image: np.ndarray,
    rng: np.random.Generator,
    group_subset: list[str] | None = None,
    num_groups_range: tuple[int, int] = GROUP_AUGMENT_COUNT_RANGE,
) -> tuple[np.ndarray, dict]:
    """Corrupt one HxWx3 float32 [0, 1] image with a random subset of the groups.

    The subset size is drawn from the inclusive num_groups_range (clamped to
    the number of selected groups), the groups are drawn without replacement
    in random order, and each chosen augmenter is built with a seed drawn from
    rng, so identical rng states produce identical images. Each step is
    clipped back to [0, 1] float32. Returns (augmented image, params dict)
    where params records the count range and one {group, augmenter, seed}
    entry per applied group, ready for JSON manifests.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.float32:
        raise ValueError(f"image must be HxWx3 float32 in [0, 1], got shape={image.shape} dtype={image.dtype}")
    low, high = normalized_group_count_range(num_groups_range=num_groups_range)
    selected = selected_group_names(group_subset=group_subset)
    num_groups = min(int(rng.integers(low=low, high=high + 1)), len(selected))
    chosen_names = [str(name) for name in rng.choice(a=selected, size=num_groups, replace=False)]
    result = image
    applied: list[dict] = []
    for group_name in chosen_names:
        factories = AUGMENTATION_GROUPS[group_name]
        factory_index = int(rng.integers(low=0, high=len(factories)))
        child_seed = int(rng.integers(low=0, high=2**31))
        augmenter = factories[factory_index](seed=child_seed)
        result = np.asarray(augmenter.augment_image(image=result))
        result = np.clip(a=result, a_min=0.0, a_max=1.0).astype(dtype=np.float32)
        applied.append({"group": group_name, "augmenter": type(augmenter).__name__, "seed": child_seed})
    params = {"count_range": (low, high), "num_groups": num_groups, "groups": applied}
    return result, params