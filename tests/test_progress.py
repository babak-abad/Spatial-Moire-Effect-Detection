"""Sanity checks for the ultralytics-style training progress bar."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import (  # noqa: E402
    PROGRESS_COLUMN_WIDTH,
    TQDM_BAR_FORMAT,
    TRAIN_PROGRESS_COLUMNS,
    VAL_PROGRESS_COLUMNS,
)
from spatial_moire.baseline.engine import run_training  # noqa: E402
from spatial_moire.baseline.progress import (  # noqa: E402
    format_progress_row,
    gpu_memory_string,
    train_header,
    val_header,
    yolo_tqdm,
)

EPOCHS = 2
NUM_SAMPLES = 24
BATCH_SIZE = 4
IMAGE_CHANNELS = 3
IMAGE_HEIGHT = 8
IMAGE_WIDTH = 8
VAL_ROW_WIDTH = PROGRESS_COLUMN_WIDTH * len(VAL_PROGRESS_COLUMNS)


def check_bar_format_matches_ultralytics() -> None:
    assert TQDM_BAR_FORMAT == "{l_bar}{bar:10}{r_bar}", "bar format must stay the ultralytics constant"
    print("ok: the tqdm bar format matches the ultralytics constant")


def check_yolo_tqdm_defaults_bar_format() -> None:
    buffer = io.StringIO()
    with yolo_tqdm(iterable=range(3), total=3, file=buffer, disable=False) as bar:
        assert bar.format_dict["bar_format"] == TQDM_BAR_FORMAT, "yolo_tqdm must default to the ultralytics format"
        for _ in bar:
            pass
    rendered = buffer.getvalue()
    assert "100%" in rendered and "| 3/3" in rendered, "the bar must render its completion frame"
    print("ok: yolo_tqdm defaults to the ultralytics bar format and renders a completion frame")


def check_header_and_value_rows_align() -> None:
    header = train_header()
    row = format_progress_row(values=(f"1/{EPOCHS}", "5.032G", "0.6931", "0.5432", 32, 224))
    assert len(header) == PROGRESS_COLUMN_WIDTH * len(TRAIN_PROGRESS_COLUMNS), "header must span the train columns"
    assert len(row) == len(header), "header and value rows must share one column grid"
    assert header[:PROGRESS_COLUMN_WIDTH].endswith("Epoch"), "the Epoch column must lead the train header"
    assert row[:PROGRESS_COLUMN_WIDTH].endswith(f"1/{EPOCHS}"), "the epoch fraction must lead the train row"
    assert len(val_header()) == VAL_ROW_WIDTH, "the val header must span the val columns"
    print("ok: header and value rows share the right-aligned column grid")


def check_gpu_memory_string_is_blank_on_cpu() -> None:
    assert gpu_memory_string(device=torch.device("cpu")) == "", "cpu runs must show a blank GPU_mem column"
    print("ok: gpu_memory_string returns a blank GPU_mem column on cpu")


def build_tiny_training_setup() -> tuple[nn.Module, DataLoader, DataLoader, torch.device]:
    torch.manual_seed(seed=0)
    features = torch.randn(size=(NUM_SAMPLES, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))
    labels = torch.randint(low=0, high=2, size=(NUM_SAMPLES,))
    train_loader = DataLoader(dataset=TensorDataset(features, labels), batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(dataset=TensorDataset(features, labels), batch_size=BATCH_SIZE, shuffle=False)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH, out_features=1),
    )
    return model, train_loader, val_loader, torch.device("cpu")


def check_run_training_prints_progress_output() -> None:
    model, train_loader, val_loader, device = build_tiny_training_setup()
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with tempfile.TemporaryDirectory() as temp_dir:
        with contextlib.redirect_stdout(new_target=stdout_buffer), contextlib.redirect_stderr(new_target=stderr_buffer):
            history = run_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                output_dir=Path(temp_dir),
                epochs=EPOCHS,
                progress=True,
            )
    output = stdout_buffer.getvalue()
    assert "Starting training for 2 epochs..." in output, "the banner must print before the first epoch"
    assert output.count(train_header()) == EPOCHS, "the train header must print once per epoch"
    val_rows = [line for line in output.splitlines() if len(line) == VAL_ROW_WIDTH and line.endswith("s")]
    assert len(val_rows) == EPOCHS, "the val metrics row must print once per epoch"
    assert len(history["epoch"]) == EPOCHS and history["best_epoch"] in range(1, EPOCHS + 1), (
        "history must stay fully populated while the bars run"
    )
    assert "100%" in stderr_buffer.getvalue(), "the batch bars must reach completion"
    print("ok: run_training prints the banner, per-epoch headers and val rows while the bars run")


def check_progress_can_be_disabled() -> None:
    model, train_loader, val_loader, device = build_tiny_training_setup()
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with tempfile.TemporaryDirectory() as temp_dir:
        with contextlib.redirect_stdout(new_target=stdout_buffer), contextlib.redirect_stderr(new_target=stderr_buffer):
            run_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                output_dir=Path(temp_dir),
                epochs=EPOCHS,
                progress=False,
            )
    assert stdout_buffer.getvalue() == "", "disabled progress must not print anything to stdout"
    assert stderr_buffer.getvalue() == "", "disabled progress must not render any bar"
    print("ok: progress=False silences bars, headers and rows entirely")


def main() -> None:
    check_bar_format_matches_ultralytics()
    check_yolo_tqdm_defaults_bar_format()
    check_header_and_value_rows_align()
    check_gpu_memory_string_is_blank_on_cpu()
    check_run_training_prints_progress_output()
    check_progress_can_be_disabled()
    print("all progress bar checks passed")


if __name__ == "__main__":
    main()

