"""Evaluate the trained ResNet-18 moire detection baseline on the held-out test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spatial_moire.baseline.data import (  # noqa: E402
    build_eval_transform,
    collect_test_samples,
    moire_dataset,
)
from spatial_moire.baseline.engine import evaluate  # noqa: E402
from spatial_moire.baseline.metrics import (  # noqa: E402
    classification_report_text,
    compute_binary_metrics,
    plot_confusion_matrix,
    save_metrics_json,
)
from spatial_moire.baseline.model import build_resnet18_classifier, load_model_checkpoint  # noqa: E402
from spatial_moire.config import (  # noqa: E402
    BEST_MODEL_FILENAME,
    CLASSIFICATION_THRESHOLD,
    CONFUSION_MATRIX_FILENAME,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    EVAL_BATCH_SIZE,
    EVAL_NUM_WORKERS,
    MOIRE_LABEL,
    PIN_MEMORY,
    TEST_METRICS_FILENAME,
    TEST_SPLIT_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the moire detection baseline on the test split.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=EVAL_NUM_WORKERS)
    parser.add_argument("--threshold", type=float, default=CLASSIFICATION_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    checkpoint_path = args.output_dir / BEST_MODEL_FILENAME
    model = build_resnet18_classifier(pretrained=False)
    checkpoint = load_model_checkpoint(model=model, checkpoint_path=checkpoint_path)
    model = model.to(device=device)
    print(f"loaded checkpoint from epoch {checkpoint['epoch']} (val F1 = {checkpoint['val_f1']:.4f})", flush=True)

    test_samples = collect_test_samples(test_dir=args.data_dir / TEST_SPLIT_DIR)
    test_positives = sum(1 for sample in test_samples if sample["label"] == MOIRE_LABEL)
    print(
        f"test: {len(test_samples)} samples ({test_positives} moire / {len(test_samples) - test_positives} clean)",
        flush=True,
    )
    test_set = moire_dataset(samples=test_samples, transform=build_eval_transform())
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
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
