"""Ultralytics-style console progress reporting for training and evaluation.

Reproduces the progress bar look of Ultralytics YOLO training: a tqdm bar
rendered with the custom ``{l_bar}{bar:10}{r_bar}`` format, a right-aligned
column header printed before each epoch's bar, and a bar description that
carries the live column values (epoch, GPU memory, running loss and
accuracy, instances, image size) while it advances.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from tqdm import tqdm

from config import PROGRESS_COLUMN_WIDTH, TQDM_BAR_FORMAT, TRAIN_PROGRESS_COLUMNS, VAL_PROGRESS_COLUMNS


class yolo_tqdm(tqdm):
    """tqdm subclass that renders the Ultralytics bar format by default."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs.setdefault("bar_format", TQDM_BAR_FORMAT)
        super().__init__(*args, **kwargs)


def format_progress_row(values: Sequence[object]) -> str:
    """Format values into right-aligned PROGRESS_COLUMN_WIDTH-wide columns like the header rows."""
    row_values = tuple(values)
    template = "%" + str(PROGRESS_COLUMN_WIDTH) + "s"
    return (template * len(row_values)) % row_values


def train_header() -> str:
    """Return the training progress header row shown above each epoch's bar."""
    return format_progress_row(values=TRAIN_PROGRESS_COLUMNS)


def val_header() -> str:
    """Return the validation progress header row used as the val bar description."""
    return format_progress_row(values=VAL_PROGRESS_COLUMNS)


def gpu_memory_string(device: torch.device) -> str:
    """Return reserved GPU memory in gigabytes (Ultralytics style) or a blank string on CPU."""
    if device.type == "cuda":
        return f"{torch.cuda.memory_reserved() / 1E9:.3f}G"
    return ""
