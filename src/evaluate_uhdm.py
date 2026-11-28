"""Evaluate a trained moire detection checkpoint on the foreign UHDM dataset.

UHDM ships paired moire / gt (clean) folders per split; the label comes from
the folder alone, exactly like the pooled-dataset contract. The UHDM pools
are never part of the training source pools, so this measures
cross-dataset (out-of-distribution) generalization.
"""

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
    collect_class_samples,
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
from config import (  # noqa: E402
    BEST_MODEL_FILENAME,
    CLASSIFICATION_THRESHOLD,
    CLEAN_LABEL,
    DEFAULT_OUTPUT_DIR,
    EVAL_BATCH_SIZE,
    EVAL_NUM_WORKERS,
    EVAL_SHUFFLE,
    MOIRE_LABEL,
    PIN_MEMORY,
    UHDM_DEFAULT_SPLIT,
    UHDM_EVAL_CONFUSION_MATRIX_FILENAME_TEMPLATE,
    UHDM_EVAL_METRICS_FILENAME_TEMPLATE,
    UHDM_EVAL_OUTPUT_DIR,
    UHDM_ROOT,
    UHDM_SPLITS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained moire detection checkpoint on the UHDM dataset.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory holding the checkpoint")
    parser.add_argument("--split", type=str, default=UHDM_DEFAULT_SPLIT, choices=list(UHDM_SPLITS))
    parser.add_argument("--output-dir", type=Path, default=UHDM_EVAL_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=EVAL_NUM_WORKERS)
    parser.add_argument("--threshold", type=float, default=CLASSIFICATION_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    checkpoint_path = args.model_dir / BEST_MODEL_FILENAME
    model = build_resnet18_classifier(pretrained=False)
    checkpoint = load_model_checkpoint(model=model, checkpoint_path=checkpoint_path)
    model = model.to(device=device)
    print(f"loaded checkpoint from epoch {checkpoint['epoch']} (val F1 = {checkpoint['val_f1']:.4f})", flush=True)

    clean_samples = collect_class_samples(folders=[UHDM_ROOT / args.split / "gt"], label=CLEAN_LABEL)
    moire_samples = collect_class_samples(folders=[UHDM_ROOT / args.split / "moire"], label=MOIRE_LABEL)
    samples = clean_samples + moire_samples
    print(
        f"UHDM {args.split}: {len(samples)} samples ({len(moire_samples)} moire / {len(clean_samples)} clean)",
        flush=True,
    )
    dataset = moire_dataset(samples=samples, transform=build_eval_transform())
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=EVAL_SHUFFLE,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY,
    )
    criterion = nn.BCEWithLogitsLoss()
    stats = evaluate(model=model, loader=loader, criterion=criterion, device=device)

    metrics = compute_binary_metrics(
        labels=stats["labels"],
        probabilities=stats["probabilities"],
        threshold=args.threshold,
    )
    metrics["dataset"] = f"uhdm_{args.split}"
    metrics["loss"] = stats["loss"]
    metrics["checkpoint_epoch"] = int(checkpoint["epoch"])
    save_metrics_json(metrics=metrics, path=args.output_dir / UHDM_EVAL_METRICS_FILENAME_TEMPLATE.format(split=args.split))
    plot_confusion_matrix(
        labels=stats["labels"],
        probabilities=stats["probabilities"],
        output_path=args.output_dir / UHDM_EVAL_CONFUSION_MATRIX_FILENAME_TEMPLATE.format(split=args.split),
        threshold=args.threshold,
    )
    print(classification_report_text(labels=stats["labels"], probabilities=stats["probabilities"], threshold=args.threshold))
    print(
        "uhdm {} | accuracy {:.4f} | precision {:.4f} | recall {:.4f} | F1 {:.4f} | ROC-AUC {:.4f}".format(
            args.split,
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