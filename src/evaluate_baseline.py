"""Evaluate the trained ResNet-18 moire detection baseline on the held-out test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spatial_moire.baseline.data import (  # noqa: E402
    SPLIT_MANIFEST_ALGORITHM,
    build_eval_transform,
    load_split_manifest,
    moire_dataset,
    split_train_val_test,
)
from spatial_moire.baseline.engine import evaluate  # noqa: E402
from spatial_moire.baseline.metrics import (  # noqa: E402
    classification_report_text,
    compute_binary_metrics,
    plot_confusion_matrix,
    save_metrics_json,
)
from spatial_moire.baseline.model import build_resnet18_classifier, load_model_checkpoint  # noqa: E402
from config import (  # noqa: E402
    BEST_MODEL_FILENAME,
    CLASSIFICATION_THRESHOLD,
    CLEAN_IMAGE_FOLDERS,
    CONFUSION_MATRIX_FILENAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    EVAL_BATCH_SIZE,
    EVAL_NUM_WORKERS,
    EVAL_SHUFFLE,
    MOIRE_IMAGE_FOLDERS,
    MOIRE_LABEL,
    PIN_MEMORY,
    SPLIT_MANIFEST_FILENAME,
    TEST_FRACTION,
    TEST_METRICS_FILENAME,
    TRAIN_FRACTION,
    VAL_FRACTION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the moire detection baseline on the test split.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=EVAL_NUM_WORKERS)
    parser.add_argument("--threshold", type=float, default=CLASSIFICATION_THRESHOLD)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the split seed (default: the seed recorded in split_manifest.json, else DEFAULT_SEED)",
    )
    return parser.parse_args()


def resolve_split_settings(
    manifest: dict[str, Any] | None,
    seed_override: int | None,
) -> tuple[int, float, float, float, str]:
    """Pick the split seed and fractions: manifest values when available, else config defaults.

    The manifest written by train_baseline.py pins the exact split that
    training held out, so its seed and fractions take precedence over the
    current config; an explicit --seed still overrides both. Returns the seed,
    the three fractions and a human-readable source description.
    """
    if manifest is not None:
        seed = int(manifest.get("seed", DEFAULT_SEED))
        train_fraction = float(manifest.get("train_fraction", TRAIN_FRACTION))
        val_fraction = float(manifest.get("val_fraction", VAL_FRACTION))
        test_fraction = float(manifest.get("test_fraction", TEST_FRACTION))
        source = SPLIT_MANIFEST_FILENAME
    else:
        seed = DEFAULT_SEED
        train_fraction = TRAIN_FRACTION
        val_fraction = VAL_FRACTION
        test_fraction = TEST_FRACTION
        source = f"config defaults (no {SPLIT_MANIFEST_FILENAME} found)"
    if seed_override is not None:
        seed = seed_override
        source = "--seed override"
    return seed, train_fraction, val_fraction, test_fraction, source


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    checkpoint_path = args.output_dir / BEST_MODEL_FILENAME
    model = build_resnet18_classifier(pretrained=False)
    checkpoint = load_model_checkpoint(model=model, checkpoint_path=checkpoint_path)
    model = model.to(device=device)
    print(f"loaded checkpoint from epoch {checkpoint['epoch']} (val F1 = {checkpoint['val_f1']:.4f})", flush=True)

    manifest = load_split_manifest(path=args.output_dir / SPLIT_MANIFEST_FILENAME)
    if manifest is not None and manifest.get("algorithm") != SPLIT_MANIFEST_ALGORITHM:
        raise ValueError(
            f"{SPLIT_MANIFEST_FILENAME} was written by split algorithm {manifest.get('algorithm')!r}; "
            f"this script only replays {SPLIT_MANIFEST_ALGORITHM!r} manifests — retrain to regenerate the split"
        )
    split_seed, train_fraction, val_fraction, test_fraction, split_source = resolve_split_settings(
        manifest=manifest,
        seed_override=args.seed,
    )
    print(
        f"test split: seed {split_seed} (from {split_source}), "
        f"train/val/test fractions {train_fraction}/{val_fraction}/{test_fraction}",
        flush=True,
    )
    _train_samples, _val_samples, test_samples = split_train_val_test(
        clean_folders=CLEAN_IMAGE_FOLDERS,
        moire_folders=MOIRE_IMAGE_FOLDERS,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=split_seed,
    )
    test_positives = sum(1 for sample in test_samples if sample["label"] == MOIRE_LABEL)
    print(
        f"test: {len(test_samples)} samples ({test_positives} moire / {len(test_samples) - test_positives} clean)",
        flush=True,
    )
    test_set = moire_dataset(samples=test_samples, transform=build_eval_transform())
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=EVAL_SHUFFLE,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY,
    )
    criterion = nn.BCEWithLogitsLoss()
    test_stats = evaluate(model=model, loader=test_loader, criterion=criterion, device=device)

    metrics = compute_binary_metrics(
        labels=test_stats["labels"],
        probabilities=test_stats["probabilities"],
        threshold=args.threshold,
    )
    metrics["test_loss"] = test_stats["loss"]
    metrics["checkpoint_epoch"] = int(checkpoint["epoch"])
    save_metrics_json(metrics=metrics, path=args.output_dir / TEST_METRICS_FILENAME)
    plot_confusion_matrix(
        labels=test_stats["labels"],
        probabilities=test_stats["probabilities"],
        output_path=args.output_dir / CONFUSION_MATRIX_FILENAME,
        threshold=args.threshold,
    )
    print(classification_report_text(labels=test_stats["labels"], probabilities=test_stats["probabilities"], threshold=args.threshold))
    print(
        "test | accuracy {:.4f} | precision {:.4f} | recall {:.4f} | F1 {:.4f} | ROC-AUC {:.4f}".format(
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics.get("roc_auc", float("nan")),
        ),
        flush=True,
    )
    print(f"outputs written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
