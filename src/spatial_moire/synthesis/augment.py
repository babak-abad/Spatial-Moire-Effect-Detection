"""Randomized moire augmentation (pattern generation + blending onto clean images)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import ndimage

from spatial_moire.synthesis.patterns import (
    alias_warp_moire,
    chromatic_fringe_pattern,
    linear_beating_pattern,
    screen_shoot_moire,
    sinusoidal_grating,
    zone_plate_pattern,
)

generator_callable = Callable[[np.ndarray, np.random.Generator], tuple[np.ndarray, dict]]


def blend_moire(image: np.ndarray, pattern: np.ndarray, strength: float) -> np.ndarray:
    """Modulate the image with a symmetric [0, 1] pattern envelope scaled by strength."""
    if pattern.ndim == 2:
        pattern = pattern[:, :, np.newaxis]
    modulated = image * (1.0 + strength * (2.0 * pattern - 1.0))
    return np.clip(a=modulated, a_min=0.0, a_max=1.0).astype(dtype=np.float32)


def random_linear_beating(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Two detuned near-Nyquist gratings multiplied into a beat pattern."""
    height, width = image.shape[:2]
    params = {
        "carrier_frequency": float(rng.uniform(low=0.32, high=0.47)),
        "carrier_angle_degrees": float(rng.uniform(low=-30.0, high=30.0)),
        "detune_frequency": float(rng.uniform(low=0.01, high=0.05)),
        "detune_angle_degrees": float(rng.uniform(low=-8.0, high=8.0)),
        "phase": float(rng.uniform(low=0.0, high=2.0 * np.pi)),
        "detune_phase": float(rng.uniform(low=0.0, high=2.0 * np.pi)),
        "strength": float(rng.uniform(low=0.35, high=0.7)),
        "blur_sigma": float(rng.uniform(low=0.0, high=0.7)),
    }
    pattern = linear_beating_pattern(
        height=height,
        width=width,
        carrier_frequency=params["carrier_frequency"],
        carrier_angle_degrees=params["carrier_angle_degrees"],
        detune_frequency=params["detune_frequency"],
        detune_angle_degrees=params["detune_angle_degrees"],
        phase=params["phase"],
        detune_phase=params["detune_phase"],
    )
    if params["blur_sigma"] > 0.0:
        pattern = ndimage.gaussian_filter(input=pattern, sigma=params["blur_sigma"])
    return blend_moire(image=image, pattern=pattern, strength=params["strength"]), params


def random_chromatic_fringe(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Per-channel sinusoids with channel-specific phase/angle offsets (color fringes)."""
    height, width = image.shape[:2]
    base_phase = float(rng.uniform(low=0.0, high=2.0 * np.pi))
    params = {
        "frequency": float(rng.uniform(low=0.004, high=0.03)),
        "angle_degrees": float(rng.uniform(low=-90.0, high=90.0)),
        "channel_phase_offsets": tuple(
            float(base_phase + channel * (2.0 * np.pi / 3.0) + rng.uniform(low=-0.3, high=0.3))
            for channel in range(3)
        ),
        "channel_angle_offsets": tuple(float(rng.uniform(low=-4.0, high=4.0)) for _ in range(3)),
        "strength": float(rng.uniform(low=0.25, high=0.55)),
    }
    pattern = chromatic_fringe_pattern(
        height=height,
        width=width,
        frequency=params["frequency"],
        angle_degrees=params["angle_degrees"],
        channel_phase_offsets=params["channel_phase_offsets"],
        channel_angle_offsets=params["channel_angle_offsets"],
    )
    return blend_moire(image=image, pattern=pattern, strength=params["strength"]), params


def random_zone_plate(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Chirped concentric rings with a random center and optional swirl."""
    height, width = image.shape[:2]
    params = {
        "ring_count": float(rng.uniform(low=10.0, high=30.0)),
        "swirl": float(rng.uniform(low=0.0, high=4.0)),
        "center_x": float(rng.uniform(low=0.25, high=0.75) * width),
        "center_y": float(rng.uniform(low=0.25, high=0.75) * height),
        "strength": float(rng.uniform(low=0.3, high=0.6)),
    }
    pattern = zone_plate_pattern(
        height=height,
        width=width,
        ring_count=params["ring_count"],
        swirl=params["swirl"],
        center=(params["center_x"], params["center_y"]),
    )
    return blend_moire(image=image, pattern=pattern, strength=params["strength"]), params


def random_screen_shoot(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Display-subpixel vs camera-grid interference simulation (color moire)."""
    direction = float(rng.choice(a=(-1.0, 1.0)))
    params = {
        "camera_rotation_degrees": float(direction * rng.uniform(low=0.4, high=5.5)),
        "camera_scale": float(
            rng.uniform(low=1.02, high=1.09) if rng.random() < 0.5 else rng.uniform(low=0.93, high=0.98)
        ),
        "offset_x": float(rng.uniform(low=-1.5, high=1.5)),
        "offset_y": float(rng.uniform(low=-1.5, high=1.5)),
        "subpixel_blur_sigma": float(rng.uniform(low=0.3, high=0.8)),
        "sensor_blur_sigma": float(rng.uniform(low=0.3, high=0.6)),
        "strength": float(rng.uniform(low=0.7, high=0.95)),
    }
    captured = screen_shoot_moire(
        image=image,
        camera_rotation_degrees=params["camera_rotation_degrees"],
        camera_scale=params["camera_scale"],
        offset_x=params["offset_x"],
        offset_y=params["offset_y"],
        subpixel_blur_sigma=params["subpixel_blur_sigma"],
        sensor_blur_sigma=params["sensor_blur_sigma"],
    )
    mixed = (1.0 - params["strength"]) * image + params["strength"] * captured
    result = np.clip(a=mixed, a_min=0.0, a_max=1.0).astype(dtype=np.float32)
    return result, params


def random_alias_warp(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Near-Nyquist colored grid warped by nearest-neighbor resampling."""
    height, width = image.shape[:2]
    direction = float(rng.choice(a=(-1.0, 1.0)))
    params = {
        "grid_pitch": float(rng.uniform(low=2.2, high=3.8)),
        "grid_angle_degrees": float(rng.uniform(low=-20.0, high=20.0)),
        "texture_strength": float(rng.uniform(low=0.4, high=0.8)),
        "scale": float(rng.uniform(low=1.02, high=1.08)),
        "rotation_degrees": float(direction * rng.uniform(low=0.5, high=3.0)),
    }
    pattern = np.empty(shape=(height, width, 3), dtype=np.float32)
    for channel in range(3):
        pattern[:, :, channel] = sinusoidal_grating(
            height=height,
            width=width,
            frequency=1.0 / params["grid_pitch"],
            angle_degrees=params["grid_angle_degrees"] + float(rng.uniform(low=-2.5, high=2.5)),
            phase=float(rng.uniform(low=0.0, high=2.0 * np.pi)),
        )
    textured = blend_moire(image=image, pattern=pattern, strength=params["texture_strength"])
    result = alias_warp_moire(
        textured_image=textured,
        scale=params["scale"],
        rotation_degrees=params["rotation_degrees"],
    )
    return result, params


MOIRE_GENERATORS: dict[str, generator_callable] = {
    "linear_beating": random_linear_beating,
    "chromatic_fringe": random_chromatic_fringe,
    "zone_plate": random_zone_plate,
    "screen_shoot": random_screen_shoot,
    "alias_warp": random_alias_warp,
}