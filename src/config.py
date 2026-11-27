"""Central configuration and constants for the Spatial-Moire-Effect-Detection project.

Single source of truth for every hard-coded value: repository paths, dataset
layout, class labels, transform / training / evaluation / DataLoader
defaults, output filenames, plotting settings, moire type prevalence
weights, pattern-engine constants, the synthesis randomization ranges and
the group-augmentation ranges.
This module lives at src/config.py next to the CLI scripts and is imported as
the top-level ``config`` module by every script, package module and test (src
is on sys.path in all entry points). It is a dependency-free leaf (it imports
only math and pathlib), so it can be imported without circular-import risk.
"""

from __future__ import annotations

import math
from pathlib import Path

# --- Repository paths ---------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = REPO_ROOT / "src"

# --- Dataset layout ------------------------------------------------------------
TRAIN_SPLIT_DIR: str = "train"
TEST_SPLIT_DIR: str = "test"
POSITIVES_DIR: str = "moire"
NEGATIVES_DIR: str = "gt"
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

DEFAULT_DATA_DIR: Path = REPO_ROOT / "datasets" / "LCDMoire" / "my_dataset_final"
DEFAULT_OUTPUT_DIR: Path = REPO_ROOT / "results" / "baseline_resnet18"
DEFAULT_SOURCE_DIR: Path = DEFAULT_DATA_DIR / TEST_SPLIT_DIR / NEGATIVES_DIR
SAMPLES_OUTPUT_DIR: Path = REPO_ROOT / "outputs"

# --- Class labels -----------------------------------------------------------------
MOIRE_LABEL: int = 1
CLEAN_LABEL: int = 0
CLASS_NAMES: tuple[str, str] = ("clean", "moire")
CLASSIFIER_OUT_FEATURES: int = 1
CLASSIFICATION_LABELS: tuple[int, int] = (CLEAN_LABEL, MOIRE_LABEL)

# --- Normalization and pixel values ---------------------------------------------------
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
PIXEL_VALUE_MAX: float = 255.0
IMAGE_COLOR_MODE: str = "RGB"

# --- Transforms -------------------------------------------------------------------------
TRAIN_CROP_SIZE: int = 224
EVAL_RESIZE_SIZE: int = 256
EVAL_CROP_SIZE: int = 224
RANDOM_RESIZED_CROP_SCALE: tuple[float, float] = (0.6, 1.0)
COLOR_JITTER_BRIGHTNESS: float = 0.2
COLOR_JITTER_CONTRAST: float = 0.2
COLOR_JITTER_SATURATION: float = 0.1
COLOR_JITTER_HUE: float = 0.02

# --- DataLoader ---------------------------------------------------------------------------
PIN_MEMORY: bool = True
TRAIN_SHUFFLE: bool = True
TRAIN_PERSISTENT_WORKERS: bool = True
VAL_PERSISTENT_WORKERS: bool = True
VAL_NUM_WORKERS_MIN: int = 2
VAL_NUM_WORKERS_DIVISOR: int = 2

# --- Training defaults ------------------------------------------------------------------------
DEFAULT_SEED: int = 42
TRAIN_EPOCHS: int = 10
TRAIN_BATCH_SIZE: int = 32
TRAIN_NUM_WORKERS: int = 8
TRAIN_PRETRAINED: bool = True
VAL_FRACTION: float = 0.2
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 1e-4
MOIRE_AUGMENT_PROB: float = 0.5

# --- Evaluation defaults --------------------------------------------------------------------------
EVAL_BATCH_SIZE: int = 64
EVAL_NUM_WORKERS: int = 4
EVAL_SHUFFLE: bool = False
CLASSIFICATION_THRESHOLD: float = 0.5

# --- Sample-generation defaults ---------------------------------------------------------------------
NUM_SAMPLE_IMAGES: int = 4
NUM_VARIANTS: int = 1
JPEG_QUALITY: int = 95
SAMPLING_MODES: tuple[str, ...] = ("grid", "weighted")
DEFAULT_SAMPLING_MODE: str = "grid"
MAX_DRAWN_PREVIEW_IMAGES: int = 8
# Format keys: type_name, source_stem, variant; the .jpg extension must stay
# in sync with JPEG_QUALITY saving.
MOIRE_SAMPLE_FILENAME_TEMPLATE: str = "moire_{type_name}__{source_stem}_v{variant:02d}.jpg"

# --- Output filenames -----------------------------------------------------------------------------------
BEST_MODEL_FILENAME: str = "best_model.pt"
HISTORY_FILENAME: str = "history.json"
TRAIN_VAL_SPLIT_FILENAME: str = "train_val_split.json"
TRAINING_CURVES_FILENAME: str = "training_curves.png"
TEST_METRICS_FILENAME: str = "test_metrics.json"
CONFUSION_MATRIX_FILENAME: str = "confusion_matrix.png"
PREVIEW_GRID_FILENAME: str = "preview_grid.png"
PREVIEW_DRAWN_FILENAME: str = "preview_drawn.png"
SYNTHESIS_MANIFEST_FILENAME: str = "moire_synthesis_params.json"

# --- Serialization -----------------------------------------------------------------------------------------
JSON_INDENT: int = 2
JSON_ENCODING: str = "utf-8"

# --- Plotting ---------------------------------------------------------------------------------------------------
MATPLOTLIB_BACKEND: str = "Agg"
PREVIEW_CELL_SIZE: float = 2.4
DRAWN_PREVIEW_ROW_HEIGHT: float = 2.6
PREVIEW_DPI: int = 110
REPORT_DPI: int = 150
PREVIEW_LABEL_FONTSIZE: int = 9
PREVIEW_TITLE_FONTSIZE: int = 10
CONFUSION_MATRIX_FIGSIZE: tuple[float, float] = (5.5, 4.5)
CONFUSION_MATRIX_ANNOTATION_FONTSIZE: int = 13
TRAINING_CURVES_FIGSIZE: tuple[float, float] = (11, 4.5)
PLOT_GRID_ALPHA: float = 0.3
SCORE_AXIS_YLIM: tuple[float, float] = (0.0, 1.05)
CLASSIFICATION_REPORT_DIGITS: int = 4

# --- Moire type prevalence weights ------------------------------------------------------------------------------
# The possibility (sampling probability) of each type is weight / total_weight
# across the active selection; mirrors docs/moire_types.md.
MOIRE_TYPE_WEIGHTS: dict[str, float] = {
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

# --- Pattern engine constants (synthesis/patterns.py) -------------------------------------------------------------
SCREEN_SHOOT_SUBPIXEL_STRIDE: int = 3
SCREEN_SHOOT_DEFAULT_SUBPIXEL_BLUR_SIGMA: float = 0.6
SCREEN_SHOOT_DEFAULT_SENSOR_BLUR_SIGMA: float = 0.4
SCREEN_SHOOT_GAIN_CLIP_RANGE: tuple[float, float] = (0.25, 4.0)
CAPTURED_MEAN_EPSILON: float = 1e-6
SCREEN_SHOOT_INTERPOLATION_ORDER: int = 1
SCREEN_SHOOT_COORDINATE_MODE: str = "nearest"
ZONE_PLATE_DEFAULT_CENTER_FRACTION: float = 0.5
ZONE_PLATE_RADIUS_MAX_FRACTION: float = 0.5

# --- Synthesis randomization ranges (synthesis/augment.py) -----------------------------------------------------------
# Every tuple is an inclusive (low, high) range drawn once with rng.uniform.
# The draw order inside augment.py follows each params dict literal order, so
# identical seeds keep producing identical images.

PHASE_RANGE: tuple[float, float] = (0.0, math.tau)
RANDOM_DIRECTIONS: tuple[float, ...] = (-1.0, 1.0)
RGB_CHANNEL_PHASE_STEP: float = math.tau / 3.0

# linear beating (pixel-grid / rotational moire)
LINEAR_BEATING_CARRIER_FREQUENCY_RANGE: tuple[float, float] = (0.32, 0.47)
LINEAR_BEATING_CARRIER_ANGLE_RANGE: tuple[float, float] = (-30.0, 30.0)
LINEAR_BEATING_DETUNE_FREQUENCY_RANGE: tuple[float, float] = (0.01, 0.05)
LINEAR_BEATING_DETUNE_ANGLE_RANGE: tuple[float, float] = (-8.0, 8.0)
LINEAR_BEATING_STRENGTH_RANGE: tuple[float, float] = (0.35, 0.7)
LINEAR_BEATING_BLUR_SIGMA_RANGE: tuple[float, float] = (0.0, 0.7)

# chromatic fringe (color moire)
CHROMATIC_FRINGE_FREQUENCY_RANGE: tuple[float, float] = (0.004, 0.03)
CHROMATIC_FRINGE_ANGLE_RANGE: tuple[float, float] = (-90.0, 90.0)
CHROMATIC_FRINGE_CHANNEL_PHASE_JITTER_RANGE: tuple[float, float] = (-0.3, 0.3)
CHROMATIC_FRINGE_CHANNEL_ANGLE_OFFSET_RANGE: tuple[float, float] = (-4.0, 4.0)
CHROMATIC_FRINGE_STRENGTH_RANGE: tuple[float, float] = (0.25, 0.55)

# zone plate (perspective moire)
ZONE_PLATE_RING_COUNT_RANGE: tuple[float, float] = (10.0, 30.0)
ZONE_PLATE_SWIRL_RANGE: tuple[float, float] = (0.0, 4.0)
ZONE_PLATE_CENTER_FRACTION_RANGE: tuple[float, float] = (0.25, 0.75)
ZONE_PLATE_STRENGTH_RANGE: tuple[float, float] = (0.3, 0.6)

# screen shoot (pixel-grid / color / rotational moire)
SCREEN_SHOOT_CAMERA_ROTATION_MAGNITUDE_RANGE: tuple[float, float] = (0.4, 5.5)
SCREEN_SHOOT_CAMERA_SCALE_UP_RANGE: tuple[float, float] = (1.02, 1.09)
SCREEN_SHOOT_CAMERA_SCALE_DOWN_RANGE: tuple[float, float] = (0.93, 0.98)
SCREEN_SHOOT_SCALE_UP_PROBABILITY: float = 0.5
SCREEN_SHOOT_OFFSET_RANGE: tuple[float, float] = (-1.5, 1.5)
SCREEN_SHOOT_SUBPIXEL_BLUR_SIGMA_RANGE: tuple[float, float] = (0.3, 0.8)
SCREEN_SHOOT_SENSOR_BLUR_SIGMA_RANGE: tuple[float, float] = (0.3, 0.6)
SCREEN_SHOOT_STRENGTH_RANGE: tuple[float, float] = (0.7, 0.95)

# alias warp (perspective / demosaic moire)
ALIAS_WARP_GRID_PITCH_RANGE: tuple[float, float] = (2.2, 3.8)
ALIAS_WARP_GRID_ANGLE_RANGE: tuple[float, float] = (-20.0, 20.0)
ALIAS_WARP_CHANNEL_ANGLE_JITTER_RANGE: tuple[float, float] = (-2.5, 2.5)
ALIAS_WARP_TEXTURE_STRENGTH_RANGE: tuple[float, float] = (0.4, 0.8)
ALIAS_WARP_SCALE_RANGE: tuple[float, float] = (1.02, 1.08)
ALIAS_WARP_ROTATION_MAGNITUDE_RANGE: tuple[float, float] = (0.5, 3.0)

# scanline bands (scanline moire)
SCANLINE_CARRIER_FREQUENCY_RANGE: tuple[float, float] = (0.32, 0.47)
SCANLINE_CARRIER_ANGLE_BASE_DEGREES: float = 90.0
SCANLINE_CARRIER_ANGLE_JITTER_RANGE: tuple[float, float] = (-3.0, 3.0)
SCANLINE_DETUNE_FREQUENCY_RANGE: tuple[float, float] = (0.01, 0.05)
SCANLINE_DETUNE_ANGLE_RANGE: tuple[float, float] = (-3.0, 3.0)
SCANLINE_STRENGTH_RANGE: tuple[float, float] = (0.3, 0.6)
SCANLINE_BLUR_SIGMA_RANGE: tuple[float, float] = (0.0, 0.7)

# temporal bands (temporal moire)
TEMPORAL_CARRIER_FREQUENCY_RANGE: tuple[float, float] = (0.35, 0.48)
TEMPORAL_CARRIER_ANGLE_BASE_DEGREES: float = 90.0
TEMPORAL_CARRIER_ANGLE_JITTER_RANGE: tuple[float, float] = (-1.5, 1.5)
TEMPORAL_DETUNE_FREQUENCY_RANGE: tuple[float, float] = (0.004, 0.02)
TEMPORAL_DETUNE_ANGLE_RANGE: tuple[float, float] = (-1.5, 1.5)
TEMPORAL_STRENGTH_RANGE: tuple[float, float] = (0.2, 0.45)
TEMPORAL_BLUR_SIGMA_RANGE: tuple[float, float] = (0.0, 0.5)

# defocus moire (defocus-modulated moire)
DEFOCUS_CARRIER_FREQUENCY_RANGE: tuple[float, float] = (0.32, 0.47)
DEFOCUS_CARRIER_ANGLE_RANGE: tuple[float, float] = (-25.0, 25.0)
DEFOCUS_DETUNE_FREQUENCY_RANGE: tuple[float, float] = (0.008, 0.04)
DEFOCUS_DETUNE_ANGLE_RANGE: tuple[float, float] = (-6.0, 6.0)
DEFOCUS_STRENGTH_RANGE: tuple[float, float] = (0.3, 0.6)
DEFOCUS_PATTERN_BLUR_SIGMA_RANGE: tuple[float, float] = (0.6, 1.8)

# --- Group augmentation ranges (synthesis/groups.py) ------------------------------------------------------------------
# Group augmentation draws, for each augmented image, a random subset of the
# named augmentation groups (noise, photometric, contrast, blur, dropouts,
# geometric), one imgaug augmenter per group, and applies them in random
# order after moire synthesis. Every child augmenter is seeded from the
# caller's rng, so identical seeds keep producing identical images. Tuples
# are inclusive (low, high) ranges drawn from the child augmenter's own rng.

GROUP_AUGMENT_COUNT_RANGE: tuple[int, int] = (2, 4)

# noise group (sensor read noise)
GROUP_NOISE_GAUSSIAN_SIGMA_RANGE: tuple[float, float] = (0.01, 0.05)
GROUP_NOISE_LAPLACE_SCALE_RANGE: tuple[float, float] = (0.01, 0.04)

# photometric group (exposure / gain)
GROUP_PHOTOMETRIC_MULTIPLY_RANGE: tuple[float, float] = (0.7, 1.3)
GROUP_PHOTOMETRIC_ADD_RANGE: tuple[float, float] = (-0.08, 0.08)

# contrast group (tone curve)
GROUP_CONTRAST_GAMMA_RANGE: tuple[float, float] = (0.7, 1.4)
GROUP_CONTRAST_LINEAR_ALPHA_RANGE: tuple[float, float] = (0.75, 1.3)
GROUP_CONTRAST_SIGMOID_GAIN_RANGE: tuple[float, float] = (3.0, 10.0)
GROUP_CONTRAST_SIGMOID_CUTOFF_RANGE: tuple[float, float] = (0.4, 0.6)

# blur group (defocus / camera shake)
GROUP_BLUR_GAUSSIAN_SIGMA_RANGE: tuple[float, float] = (0.0, 1.2)
GROUP_BLUR_AVERAGE_KERNEL_RANGE: tuple[int, int] = (3, 5)
GROUP_BLUR_MOTION_KERNEL: int = 7
GROUP_BLUR_MOTION_ANGLE_RANGE: tuple[float, float] = (-45.0, 45.0)

# dropouts group (dead pixels / lost packets)
GROUP_DROPOUT_COARSE_PROBABILITY_RANGE: tuple[float, float] = (0.0, 0.05)
GROUP_DROPOUT_COARSE_SIZE_PERCENT_RANGE: tuple[float, float] = (0.1, 0.3)
GROUP_DROPOUT_PIXEL_PROBABILITY_RANGE: tuple[float, float] = (0.0, 0.02)

# geometric group (camera angle drift, constant fill)
GROUP_GEOMETRIC_SCALE_RANGE: tuple[float, float] = (0.92, 1.08)
GROUP_GEOMETRIC_ROTATION_RANGE: tuple[float, float] = (-8.0, 8.0)
GROUP_GEOMETRIC_SHEAR_RANGE: tuple[float, float] = (-6.0, 6.0)
GROUP_GEOMETRIC_TRANSLATE_PERCENT_RANGE: tuple[float, float] = (0.0, 0.04)
GROUP_GEOMETRIC_FILL_VALUE: float = 0.5
DEFOCUS_IMAGE_BLUR_SIGMA_RANGE: tuple[float, float] = (0.3, 0.9)