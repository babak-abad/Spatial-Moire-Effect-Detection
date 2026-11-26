"""Procedural moire synthesis: pattern builders and randomized augmentation."""

from spatial_moire.synthesis.augment import MOIRE_GENERATORS, blend_moire

__all__ = ["MOIRE_GENERATORS", "blend_moire"]