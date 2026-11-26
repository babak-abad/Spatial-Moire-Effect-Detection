"""Procedural moire pattern builders (numpy / scipy / PIL).

Mechanisms are inspired by classic geometric moire theory (two-layer
interference), the synthetic LCDMoire dataset (arXiv 1911.02498) and
UniDemoire's pattern-then-blend synthesis pipeline (arXiv 2502.06324).
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image
from scipy import ndimage

from spatial_moire.config import (
    CAPTURED_MEAN_EPSILON,
    PIXEL_VALUE_MAX,
    SCREEN_SHOOT_DEFAULT_SENSOR_BLUR_SIGMA,
    SCREEN_SHOOT_DEFAULT_SUBPIXEL_BLUR_SIGMA,
    SCREEN_SHOOT_GAIN_CLIP_RANGE,
    SCREEN_SHOOT_SUBPIXEL_STRIDE,
)


def coordinate_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    grids = np.mgrid[0:height, 0:width].astype(dtype=np.float32)
    return grids[0], grids[1]


def sinusoidal_grating(
    height: int,
    width: int,
    frequency: float,
    angle_degrees: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Return a [0, 1] sinusoidal grating, frequency in cycles per pixel."""
    ys, xs = coordinate_grid(height=height, width=width)
    angle = math.radians(angle_degrees)
    projection = xs * math.cos(angle) + ys * math.sin(angle)
    modulation = 2.0 * np.pi * frequency * projection + phase
    return (0.5 + 0.5 * np.cos(modulation)).astype(dtype=np.float32)


def linear_beating_pattern(
    height: int,
    width: int,
    carrier_frequency: float,
    carrier_angle_degrees: float,
    detune_frequency: float,
    detune_angle_degrees: float,
    phase: float = 0.0,
    detune_phase: float = 0.0,
) -> np.ndarray:
    """Multiply two near-Nyquist gratings; their detuning beats into low-frequency bands."""
    layer_a = sinusoidal_grating(
        height=height,
        width=width,
        frequency=carrier_frequency,
        angle_degrees=carrier_angle_degrees,
        phase=phase,
    )
    layer_b = sinusoidal_grating(
        height=height,
        width=width,
        frequency=carrier_frequency - detune_frequency,
        angle_degrees=carrier_angle_degrees + detune_angle_degrees,
        phase=detune_phase,
    )
    return layer_a * layer_b


def chromatic_fringe_pattern(
    height: int,
    width: int,
    frequency: float,
    angle_degrees: float,
    channel_phase_offsets: tuple[float, float, float],
    channel_angle_offsets: tuple[float, float, float],
) -> np.ndarray:
    """Return an (H, W, 3) pattern of per-channel sinusoids that read as color fringes."""
    pattern = np.empty(shape=(height, width, 3), dtype=np.float32)
    for channel in range(3):
        pattern[:, :, channel] = sinusoidal_grating(
            height=height,
            width=width,
            frequency=frequency,
            angle_degrees=angle_degrees + channel_angle_offsets[channel],
            phase=channel_phase_offsets[channel],
        )
    return pattern


def zone_plate_pattern(
    height: int,
    width: int,
    ring_count: float,
    swirl: float = 0.0,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Chirped concentric rings with an optional angular swirl (spiral moire)."""
    ys, xs = coordinate_grid(height=height, width=width)
    if center is None:
        center = (width * 0.5, height * 0.5)
    dx = xs - center[0]
    dy = ys - center[1]
    radius = np.hypot(dx, dy)
    radius_max = np.hypot(width, height) * 0.5
    theta = np.arctan2(dy, dx)
    phase = 2.0 * np.pi * ring_count * (radius / radius_max) ** 2 + swirl * theta
    return (0.5 + 0.5 * np.cos(phase)).astype(dtype=np.float32)


def screen_shoot_moire(
    image: np.ndarray,
    camera_rotation_degrees: float,
    camera_scale: float,
    offset_x: float,
    offset_y: float,
    subpixel_blur_sigma: float = SCREEN_SHOOT_DEFAULT_SUBPIXEL_BLUR_SIGMA,
    sensor_blur_sigma: float = SCREEN_SHOOT_DEFAULT_SENSOR_BLUR_SIGMA,
) -> np.ndarray:
    """Simulate photographing the image on an RGB-stripe LCD with a misaligned camera.

    A high-resolution display buffer holds one R/G/B subpixel column per display
    pixel; sampling it on the rotated/scaled camera grid beats the subpixel
    stripes into color moire.
    """
    height, width = image.shape[:2]
    display = np.zeros(shape=(height, width * 3, 3), dtype=np.float32)
    display[:, 0::3, 0] = image[:, :, 0]
    display[:, 1::3, 1] = image[:, :, 1]
    display[:, 2::3, 2] = image[:, :, 2]
    if subpixel_blur_sigma > 0.0:
        display = ndimage.gaussian_filter(input=display, sigma=(0.0, subpixel_blur_sigma, 0.0))

    ys, xs = coordinate_grid(height=height, width=width)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    dx = xs - center_x
    dy = ys - center_y
    angle = math.radians(camera_rotation_degrees)
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    pitch = SCREEN_SHOOT_SUBPIXEL_STRIDE * camera_scale
    columns = (SCREEN_SHOOT_SUBPIXEL_STRIDE * center_x + offset_x) + pitch * (cos_angle * dx - sin_angle * dy)
    rows = (center_y + offset_y) + pitch * (sin_angle * dx + cos_angle * dy)
    coordinates = [rows.ravel(), columns.ravel()]
    captured = np.empty(shape=(height, width, 3), dtype=np.float32)
    for channel in range(3):
        sampled = ndimage.map_coordinates(
            input=display[:, :, channel],
            coordinates=coordinates,
            order=1,
            mode="nearest",
        )
        captured[:, :, channel] = sampled.reshape(height, width)
    if sensor_blur_sigma > 0.0:
        captured = ndimage.gaussian_filter(
            input=captured,
            sigma=(sensor_blur_sigma, sensor_blur_sigma, 0.0),
        )
    source_means = image.mean(axis=(0, 1), keepdims=True)
    captured_means = captured.mean(axis=(0, 1), keepdims=True)
    gains = np.clip(
        a=source_means / np.maximum(captured_means, CAPTURED_MEAN_EPSILON),
        a_min=SCREEN_SHOOT_GAIN_CLIP_RANGE[0],
        a_max=SCREEN_SHOOT_GAIN_CLIP_RANGE[1],
    )
    return np.clip(a=captured * gains, a_min=0.0, a_max=1.0).astype(dtype=np.float32)


def alias_warp_moire(
    textured_image: np.ndarray,
    scale: float,
    rotation_degrees: float,
) -> np.ndarray:
    """Rotate/scale a near-Nyquist textured image with nearest-neighbor resampling.

    Aliasing between the fine texture and the resampling grid produces moire
    beat bands.
    """
    height, width = textured_image.shape[:2]
    array = (np.clip(a=textured_image, a_min=0.0, a_max=1.0) * PIXEL_VALUE_MAX).round().astype(dtype=np.uint8)
    fill = tuple(int(value) for value in array.mean(axis=(0, 1)).round())
    pil_image = Image.fromarray(obj=array)
    pil_image = pil_image.rotate(
        angle=rotation_degrees,
        resample=Image.Resampling.NEAREST,
        expand=False,
        fillcolor=fill,
    )
    new_size = (int(round(width * scale)), int(round(height * scale)))
    pil_image = pil_image.resize(size=new_size, resample=Image.Resampling.NEAREST)
    left = (new_size[0] - width) // 2
    top = (new_size[1] - height) // 2
    pil_image = pil_image.crop(box=(left, top, left + width, top + height))
    return np.asarray(pil_image, dtype=np.float32) / PIXEL_VALUE_MAX