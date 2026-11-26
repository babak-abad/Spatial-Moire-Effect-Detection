"""Randomized moire augmentation (pattern generation + blending onto clean images)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import ndimage

from spatial_moire.config import (
    ALIAS_WARP_CHANNEL_ANGLE_JITTER_RANGE,
    ALIAS_WARP_GRID_ANGLE_RANGE,
    ALIAS_WARP_GRID_PITCH_RANGE,
    ALIAS_WARP_ROTATION_MAGNITUDE_RANGE,
    ALIAS_WARP_SCALE_RANGE,
    ALIAS_WARP_TEXTURE_STRENGTH_RANGE,
    CHROMATIC_FRINGE_ANGLE_RANGE,
    CHROMATIC_FRINGE_CHANNEL_ANGLE_OFFSET_RANGE,
    CHROMATIC_FRINGE_CHANNEL_PHASE_JITTER_RANGE,
    CHROMATIC_FRINGE_FREQUENCY_RANGE,
    CHROMATIC_FRINGE_STRENGTH_RANGE,
    DEFOCUS_CARRIER_ANGLE_RANGE,
    DEFOCUS_CARRIER_FREQUENCY_RANGE,
    DEFOCUS_DETUNE_ANGLE_RANGE,
    DEFOCUS_DETUNE_FREQUENCY_RANGE,
    DEFOCUS_IMAGE_BLUR_SIGMA_RANGE,
    DEFOCUS_PATTERN_BLUR_SIGMA_RANGE,
    DEFOCUS_STRENGTH_RANGE,
    LINEAR_BEATING_BLUR_SIGMA_RANGE,
    LINEAR_BEATING_CARRIER_ANGLE_RANGE,
    LINEAR_BEATING_CARRIER_FREQUENCY_RANGE,
    LINEAR_BEATING_DETUNE_ANGLE_RANGE,
    LINEAR_BEATING_DETUNE_FREQUENCY_RANGE,
    LINEAR_BEATING_STRENGTH_RANGE,
    PHASE_RANGE,
    RANDOM_DIRECTIONS,
    RGB_CHANNEL_PHASE_STEP,
    SCANLINE_BLUR_SIGMA_RANGE,
    SCANLINE_CARRIER_ANGLE_BASE_DEGREES,
    SCANLINE_CARRIER_ANGLE_JITTER_RANGE,
    SCANLINE_CARRIER_FREQUENCY_RANGE,
    SCANLINE_DETUNE_ANGLE_RANGE,
    SCANLINE_DETUNE_FREQUENCY_RANGE,
    SCANLINE_STRENGTH_RANGE,
    SCREEN_SHOOT_CAMERA_ROTATION_MAGNITUDE_RANGE,
    SCREEN_SHOOT_CAMERA_SCALE_DOWN_RANGE,
    SCREEN_SHOOT_CAMERA_SCALE_UP_RANGE,
    SCREEN_SHOOT_OFFSET_RANGE,
    SCREEN_SHOOT_SCALE_UP_PROBABILITY,
    SCREEN_SHOOT_SENSOR_BLUR_SIGMA_RANGE,
    SCREEN_SHOOT_STRENGTH_RANGE,
    SCREEN_SHOOT_SUBPIXEL_BLUR_SIGMA_RANGE,
    TEMPORAL_BLUR_SIGMA_RANGE,
    TEMPORAL_CARRIER_ANGLE_BASE_DEGREES,
    TEMPORAL_CARRIER_ANGLE_JITTER_RANGE,
    TEMPORAL_CARRIER_FREQUENCY_RANGE,
    TEMPORAL_DETUNE_ANGLE_RANGE,
    TEMPORAL_DETUNE_FREQUENCY_RANGE,
    TEMPORAL_STRENGTH_RANGE,
    ZONE_PLATE_CENTER_FRACTION_RANGE,
    ZONE_PLATE_RING_COUNT_RANGE,
    ZONE_PLATE_STRENGTH_RANGE,
    ZONE_PLATE_SWIRL_RANGE,
)
from spatial_moire.synthesis.patterns import (
    alias_warp_moire,
    chromatic_fringe_pattern,
    linear_beating_pattern,
    screen_shoot_moire,
    sinusoidal_grating,
    zone_plate_pattern,
)

generator_callable = Callable[[np.ndarray, np.random.Generator], tuple[np.ndarray, dict]]


def random_range_value(rng: np.random.Generator, value_range: tuple[float, float]) -> float:
    """Draw one uniform sample from an inclusive (low, high) config range."""
    return float(rng.uniform(low=value_range[0], high=value_range[1]))


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
        "carrier_frequency": random_range_value(rng=rng, value_range=LINEAR_BEATING_CARRIER_FREQUENCY_RANGE),
        "carrier_angle_degrees": random_range_value(rng=rng, value_range=LINEAR_BEATING_CARRIER_ANGLE_RANGE),
        "detune_frequency": random_range_value(rng=rng, value_range=LINEAR_BEATING_DETUNE_FREQUENCY_RANGE),
        "detune_angle_degrees": random_range_value(rng=rng, value_range=LINEAR_BEATING_DETUNE_ANGLE_RANGE),
        "phase": random_range_value(rng=rng, value_range=PHASE_RANGE),
        "detune_phase": random_range_value(rng=rng, value_range=PHASE_RANGE),
        "strength": random_range_value(rng=rng, value_range=LINEAR_BEATING_STRENGTH_RANGE),
        "blur_sigma": random_range_value(rng=rng, value_range=LINEAR_BEATING_BLUR_SIGMA_RANGE),
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
    base_phase = random_range_value(rng=rng, value_range=PHASE_RANGE)
    params = {
        "frequency": random_range_value(rng=rng, value_range=CHROMATIC_FRINGE_FREQUENCY_RANGE),
        "angle_degrees": random_range_value(rng=rng, value_range=CHROMATIC_FRINGE_ANGLE_RANGE),
        "channel_phase_offsets": tuple(
            float(
                base_phase
                + channel * RGB_CHANNEL_PHASE_STEP
                + random_range_value(rng=rng, value_range=CHROMATIC_FRINGE_CHANNEL_PHASE_JITTER_RANGE)
            )
            for channel in range(3)
        ),
        "channel_angle_offsets": tuple(
            random_range_value(rng=rng, value_range=CHROMATIC_FRINGE_CHANNEL_ANGLE_OFFSET_RANGE) for _ in range(3)
        ),
        "strength": random_range_value(rng=rng, value_range=CHROMATIC_FRINGE_STRENGTH_RANGE),
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
        "ring_count": random_range_value(rng=rng, value_range=ZONE_PLATE_RING_COUNT_RANGE),
        "swirl": random_range_value(rng=rng, value_range=ZONE_PLATE_SWIRL_RANGE),
        "center_x": random_range_value(rng=rng, value_range=ZONE_PLATE_CENTER_FRACTION_RANGE) * width,
        "center_y": random_range_value(rng=rng, value_range=ZONE_PLATE_CENTER_FRACTION_RANGE) * height,
        "strength": random_range_value(rng=rng, value_range=ZONE_PLATE_STRENGTH_RANGE),
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
    direction = float(rng.choice(a=RANDOM_DIRECTIONS))
    params = {
        "camera_rotation_degrees": float(
            direction * random_range_value(rng=rng, value_range=SCREEN_SHOOT_CAMERA_ROTATION_MAGNITUDE_RANGE)
        ),
        "camera_scale": random_range_value(
            rng=rng,
            value_range=(
                SCREEN_SHOOT_CAMERA_SCALE_UP_RANGE
                if rng.random() < SCREEN_SHOOT_SCALE_UP_PROBABILITY
                else SCREEN_SHOOT_CAMERA_SCALE_DOWN_RANGE
            ),
        ),
        "offset_x": random_range_value(rng=rng, value_range=SCREEN_SHOOT_OFFSET_RANGE),
        "offset_y": random_range_value(rng=rng, value_range=SCREEN_SHOOT_OFFSET_RANGE),
        "subpixel_blur_sigma": random_range_value(rng=rng, value_range=SCREEN_SHOOT_SUBPIXEL_BLUR_SIGMA_RANGE),
        "sensor_blur_sigma": random_range_value(rng=rng, value_range=SCREEN_SHOOT_SENSOR_BLUR_SIGMA_RANGE),
        "strength": random_range_value(rng=rng, value_range=SCREEN_SHOOT_STRENGTH_RANGE),
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
    direction = float(rng.choice(a=RANDOM_DIRECTIONS))
    params = {
        "grid_pitch": random_range_value(rng=rng, value_range=ALIAS_WARP_GRID_PITCH_RANGE),
        "grid_angle_degrees": random_range_value(rng=rng, value_range=ALIAS_WARP_GRID_ANGLE_RANGE),
        "texture_strength": random_range_value(rng=rng, value_range=ALIAS_WARP_TEXTURE_STRENGTH_RANGE),
        "scale": random_range_value(rng=rng, value_range=ALIAS_WARP_SCALE_RANGE),
        "rotation_degrees": float(
            direction * random_range_value(rng=rng, value_range=ALIAS_WARP_ROTATION_MAGNITUDE_RANGE)
        ),
    }
    pattern = np.empty(shape=(height, width, 3), dtype=np.float32)
    for channel in range(3):
        pattern[:, :, channel] = sinusoidal_grating(
            height=height,
            width=width,
            frequency=1.0 / params["grid_pitch"],
            angle_degrees=(
                params["grid_angle_degrees"]
                + random_range_value(rng=rng, value_range=ALIAS_WARP_CHANNEL_ANGLE_JITTER_RANGE)
            ),
            phase=random_range_value(rng=rng, value_range=PHASE_RANGE),
        )
    textured = blend_moire(image=image, pattern=pattern, strength=params["texture_strength"])
    result = alias_warp_moire(
        textured_image=textured,
        scale=params["scale"],
        rotation_degrees=params["rotation_degrees"],
    )
    return result, params


def random_scanline_bands(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Near-horizontal detuned gratings beating into thick venetian-blind bars."""
    height, width = image.shape[:2]
    params = {
        "carrier_frequency": random_range_value(rng=rng, value_range=SCANLINE_CARRIER_FREQUENCY_RANGE),
        "carrier_angle_degrees": (
            SCANLINE_CARRIER_ANGLE_BASE_DEGREES
            + random_range_value(rng=rng, value_range=SCANLINE_CARRIER_ANGLE_JITTER_RANGE)
        ),
        "detune_frequency": random_range_value(rng=rng, value_range=SCANLINE_DETUNE_FREQUENCY_RANGE),
        "detune_angle_degrees": random_range_value(rng=rng, value_range=SCANLINE_DETUNE_ANGLE_RANGE),
        "phase": random_range_value(rng=rng, value_range=PHASE_RANGE),
        "detune_phase": random_range_value(rng=rng, value_range=PHASE_RANGE),
        "strength": random_range_value(rng=rng, value_range=SCANLINE_STRENGTH_RANGE),
        "blur_sigma": random_range_value(rng=rng, value_range=SCANLINE_BLUR_SIGMA_RANGE),
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


def random_temporal_bands(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Luminance-only horizontal bands (rolling-shutter / PWM beat, single-frame view)."""
    height, width = image.shape[:2]
    params = {
        "carrier_frequency": random_range_value(rng=rng, value_range=TEMPORAL_CARRIER_FREQUENCY_RANGE),
        "carrier_angle_degrees": (
            TEMPORAL_CARRIER_ANGLE_BASE_DEGREES
            + random_range_value(rng=rng, value_range=TEMPORAL_CARRIER_ANGLE_JITTER_RANGE)
        ),
        "detune_frequency": random_range_value(rng=rng, value_range=TEMPORAL_DETUNE_FREQUENCY_RANGE),
        "detune_angle_degrees": random_range_value(rng=rng, value_range=TEMPORAL_DETUNE_ANGLE_RANGE),
        "phase": random_range_value(rng=rng, value_range=PHASE_RANGE),
        "detune_phase": random_range_value(rng=rng, value_range=PHASE_RANGE),
        "strength": random_range_value(rng=rng, value_range=TEMPORAL_STRENGTH_RANGE),
        "blur_sigma": random_range_value(rng=rng, value_range=TEMPORAL_BLUR_SIGMA_RANGE),
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


def random_defocus_moire(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Beat pattern washed by a strong low-pass blur (focus-dependent visibility)."""
    height, width = image.shape[:2]
    params = {
        "carrier_frequency": random_range_value(rng=rng, value_range=DEFOCUS_CARRIER_FREQUENCY_RANGE),
        "carrier_angle_degrees": random_range_value(rng=rng, value_range=DEFOCUS_CARRIER_ANGLE_RANGE),
        "detune_frequency": random_range_value(rng=rng, value_range=DEFOCUS_DETUNE_FREQUENCY_RANGE),
        "detune_angle_degrees": random_range_value(rng=rng, value_range=DEFOCUS_DETUNE_ANGLE_RANGE),
        "phase": random_range_value(rng=rng, value_range=PHASE_RANGE),
        "detune_phase": random_range_value(rng=rng, value_range=PHASE_RANGE),
        "strength": random_range_value(rng=rng, value_range=DEFOCUS_STRENGTH_RANGE),
        "pattern_blur_sigma": random_range_value(rng=rng, value_range=DEFOCUS_PATTERN_BLUR_SIGMA_RANGE),
        "image_blur_sigma": random_range_value(rng=rng, value_range=DEFOCUS_IMAGE_BLUR_SIGMA_RANGE),
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
    pattern = ndimage.gaussian_filter(input=pattern, sigma=params["pattern_blur_sigma"])
    result = blend_moire(image=image, pattern=pattern, strength=params["strength"])
    if params["image_blur_sigma"] > 0.0:
        result = ndimage.gaussian_filter(
            input=result,
            sigma=(params["image_blur_sigma"], params["image_blur_sigma"], 0.0),
        )
    return np.clip(a=result, a_min=0.0, a_max=1.0).astype(dtype=np.float32), params


def random_nested_moire(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Two different generators applied in sequence (coarse + fine superimposed moire)."""
    candidates = sorted(name for name in MOIRE_GENERATORS if name != "nested_moire")
    primary_name, secondary_name = (str(name) for name in rng.choice(a=candidates, size=2, replace=False))
    primary_image, primary_params = MOIRE_GENERATORS[primary_name](image=image, rng=rng)
    nested_image, secondary_params = MOIRE_GENERATORS[secondary_name](image=primary_image, rng=rng)
    params = {
        "primary_generator": primary_name,
        "primary_params": primary_params,
        "secondary_generator": secondary_name,
        "secondary_params": secondary_params,
    }
    return nested_image, params


MOIRE_GENERATORS: dict[str, generator_callable] = {
    "linear_beating": random_linear_beating,
    "chromatic_fringe": random_chromatic_fringe,
    "zone_plate": random_zone_plate,
    "screen_shoot": random_screen_shoot,
    "alias_warp": random_alias_warp,
    "scanline_bands": random_scanline_bands,
    "temporal_bands": random_temporal_bands,
    "defocus_moire": random_defocus_moire,
    "nested_moire": random_nested_moire,
}