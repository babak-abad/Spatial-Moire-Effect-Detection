"""Weighted registry of moire effect types.

Each type documents its physical cause and visual signature and carries a
prevalence weight (values live in spatial_moire.config.MOIRE_TYPE_WEIGHTS).
The possibility (sampling probability) of each type is
weight / total_weight across the active selection, mirroring
docs/moire_types.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spatial_moire.config import MOIRE_TYPE_WEIGHTS


@dataclass(frozen=True)
class moire_type_info:
    """One moire effect type: documentation plus prevalence weight and synthesis mapping."""

    name: str
    title: str
    category: str
    cause: str
    appearance: str
    weight: float
    generator_names: tuple[str, ...]


MOIRE_TYPES: dict[str, moire_type_info] = {
    "pixel_grid": moire_type_info(
        name="pixel_grid",
        title="Pixel-grid moire (screen-door / honeycomb)",
        category="grid_interaction",
        cause="screen pixel pitch projected onto the sensor roughly equals the sensor pixel pitch (beat frequency)",
        appearance="coarse wavy bands, worm-like or cellular blobs, honeycomb lattices, a faint floating watermark",
        weight=MOIRE_TYPE_WEIGHTS["pixel_grid"],
        generator_names=("linear_beating", "screen_shoot"),
    ),
    "color": moire_type_info(
        name="color",
        title="Color moire (rainbow / false-color)",
        category="grid_interaction",
        cause="LCD RGB subpixel stripes alias against the camera Bayer CFA, which samples R, G and B apart",
        appearance="magenta/green or cyan/red fringes, rainbow blotches, colored bands that shift with camera motion",
        weight=MOIRE_TYPE_WEIGHTS["color"],
        generator_names=("chromatic_fringe", "screen_shoot"),
    ),
    "scanline": moire_type_info(
        name="scanline",
        title="Scan-line moire (CRT / interlaced content)",
        category="display_technology",
        cause="horizontal scanlines or interlaced fields beat against the camera sensor rows",
        appearance="horizontal banding, thick venetian-blind bars, slow vertical drift; curved bands on spherical CRTs",
        weight=MOIRE_TYPE_WEIGHTS["scanline"],
        generator_names=("scanline_bands",),
    ),
    "temporal": moire_type_info(
        name="temporal",
        title="Temporal moire (flicker / rolling bands)",
        category="time_domain",
        cause="refresh rate or PWM backlight dimming beats against shutter speed and rolling-shutter readout",
        appearance="horizontal brightness bands rolling up or down, moving bars on video, dark/light stripes on stills",
        weight=MOIRE_TYPE_WEIGHTS["temporal"],
        generator_names=("temporal_bands",),
    ),
    "perspective": moire_type_info(
        name="perspective",
        title="Perspective / keystone moire (chirped)",
        category="geometry",
        cause="an oblique shooting angle makes the projected screen frequency vary across the frame (chirp)",
        appearance="curving, converging fringes: contour-line arcs, hyperbolic shapes, compression toward one side",
        weight=MOIRE_TYPE_WEIGHTS["perspective"],
        generator_names=("zone_plate", "alias_warp"),
    ),
    "rotational": moire_type_info(
        name="rotational",
        title="Rotational moire",
        category="geometry",
        cause="the camera is rotated slightly (even 1-2 degrees) relative to the screen pixel axes",
        appearance="rotated fringe sets; curved or hyperbolic fringes with lens distortion, the swirly monitor look",
        weight=MOIRE_TYPE_WEIGHTS["rotational"],
        generator_names=("linear_beating", "screen_shoot"),
    ),
    "defocus": moire_type_info(
        name="defocus",
        title="Defocus-modulated moire",
        category="modulation",
        cause="slight blur acts as a low-pass filter on the screen grid before it reaches the sensor",
        appearance="moire that appears, shifts or vanishes while racking focus; a diagnostic for sampling moire",
        weight=MOIRE_TYPE_WEIGHTS["defocus"],
        generator_names=("defocus_moire",),
    ),
    "demosaic": moire_type_info(
        name="demosaic",
        title="Demosaic artifacts (moire's companions)",
        category="companion",
        cause="the demosaic algorithm guesses color from an aliased Bayer pattern",
        appearance="zipper or maze patterns, false color at edges, cross-hatching riding along with color moire",
        weight=MOIRE_TYPE_WEIGHTS["demosaic"],
        generator_names=("alias_warp",),
    ),
    "nested": moire_type_info(
        name="nested",
        title="Nested / recursive moire",
        category="recursive",
        cause="the screen displays content that itself has periodic structure (photo of a screen, stripes, halftone)",
        appearance="two superimposed moire patterns: a coarse one from the screen grid, a finer one from the content",
        weight=MOIRE_TYPE_WEIGHTS["nested"],
        generator_names=("nested_moire",),
    ),
}


def _selected_type_names(type_names: list[str] | None) -> list[str]:
    selected = list(MOIRE_TYPES) if type_names is None else list(type_names)
    unknown_names = [name for name in selected if name not in MOIRE_TYPES]
    if unknown_names:
        raise ValueError(f"unknown moire type names: {unknown_names}; expected one of {sorted(MOIRE_TYPES)}")
    if not selected:
        raise ValueError("at least one moire type must be selected")
    return selected


def total_weight(type_names: list[str] | None = None) -> float:
    """Sum of the weights of the selected moire types (all types by default)."""
    selected = _selected_type_names(type_names=type_names)
    return float(sum(MOIRE_TYPES[name].weight for name in selected))


def type_probabilities(type_names: list[str] | None = None) -> dict[str, float]:
    """Possibility of each selected type: weight / total_weight; the values sum to 1.0."""
    selected = _selected_type_names(type_names=type_names)
    total = total_weight(type_names=selected)
    probabilities = {name: MOIRE_TYPES[name].weight / total for name in selected}
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("moire type possibilities must sum to 1.0")
    return probabilities


def sample_moire_type(rng: np.random.Generator, type_names: list[str] | None = None) -> str:
    """Draw one moire type name with possibility weight / total_weight."""
    selected = _selected_type_names(type_names=type_names)
    weights = np.array(object=[MOIRE_TYPES[name].weight for name in selected], dtype=np.float64)
    return str(rng.choice(a=selected, p=weights / weights.sum()))


def generator_names_for_type(type_name: str) -> tuple[str, ...]:
    """Synthesis generator names registered for one moire type."""
    if type_name not in MOIRE_TYPES:
        raise ValueError(f"unknown moire type: {type_name!r}; expected one of {sorted(MOIRE_TYPES)}")
    return MOIRE_TYPES[type_name].generator_names