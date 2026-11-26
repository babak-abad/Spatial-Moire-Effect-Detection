"""Fine-tune an ImageNet-pretrained ResNet-18 as the deep-learning baseline for moire detection."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spatial_moire.baseline.augmentation import augmented_moire_dataset  # noqa: E402
from spatial_moire.baseline.data import (  # noqa: E402
    build_eval_transform,
    build_train_transform,
    moire_dataset,
    save_split_manifest,
    split_train_val,
)
from spatial_moire.baseline.engine import run_training  # noqa: E402
from spatial_moire.baseline.metrics import plot_training_history  # noqa: E402
from spatial_moire.baseline.model import build_resnet18_classifier  # noqa: E402
from config import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    HISTORY_FILENAME,
    JSON_ENCODING,
    JSON_INDENT,
    LEARNING_RATE,
    MOIRE_AUGMENT_PROB,
    MOIRE_LABEL,
    PIN_MEMORY,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_NUM_WORKERS,
    TRAIN_PERSISTENT_WORKERS,
    TRAIN_PRETRAINED,
    TRAIN_SHUFFLE,
    TRAINING_CURVES_FILENAME,
    TRAIN_SPLIT_DIR,
    TRAIN_VAL_SPLIT_FILENAME,
    VAL_FRACTION,
    VAL_NUM_WORKERS_DIVISOR,
    VAL_NUM_WORKERS_MIN,
    VAL_PERSISTENT_WORKERS,
    WEIGHT_DECAY,
)
from spatial_moire.moire_types import MOIRE_TYPES, type_probabilities  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ResNet-18 moire detection baseline on the local LCDMoire subset.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--num-workers", type=int, default=TRAIN_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--moire-augment-prob", type=float, default=MOIRE_AUGMENT_PROB)
    parser.add_argument(
        "--moire-augment-types",
        nargs="+",
        choices=sorted(MOIRE_TYPES),
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(device=device)}", flush=True)

    train_samples, val_samples = split_train_val(
        train_dir=args.data_dir / TRAIN_SPLIT_DIR,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    train_positives = sum(1 for sample in train_samples if sample["label"] == MOIRE_LABEL)
    val_positives = sum(1 for sample in val_samples if sample["label"] == MOIRE_LABEL)
    print(
        f"train: {len(train_samples)} samples ({train_positives} moire / {len(train_samples) - train_positives} clean)",
        flush=True,
    )
    print(
        f"val: {len(val_samples)} samples ({val_positives} moire / {len(val_samples) - val_positives} clean)",
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_split_manifest(
        train_samples=train_samples,
        val_samples=val_samples,
        path=args.output_dir / TRAIN_VAL_SPLIT_FILENAME,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )

    train_type_names = list(MOIRE_TYPES) if args.moire_augment_types is None else sorted(args.moire_augment_types)
    if args.moire_augment_prob > 0.0:
        train_set = augmented_moire_dataset(
            samples=train_samples,
            transform=build_train_transform(),
            augment_prob=args.moire_augment_prob,
            type_names=train_type_names,
            seed=args.seed,
        )
        expected_moire_share = (
            train_positives + args.moire_augment_prob * (len(train_samples) - train_positives)
        ) / len(train_samples)
        type_mix = ", ".join(
            f"{name} {probability:.0%}"
            for name, probability in type_probabilities(type_names=train_type_names).items()
        )
        print(
            f"moire augmentation: clean train images augmented on the fly with probability {args.moire_augment_prob} "
            f"via weighted moire type draw ({type_mix}) and relabeled as moire "
            f"(expected moire share ~{expected_moire_share:.0%})",
            flush=True,
        )
    else:
        train_set = moire_dataset(samples=train_samples, transform=build_train_transform())
    val_set = moire_dataset(samples=val_samples, transform=build_eval_transform())
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=TRAIN_SHUFFLE,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=args.num_workers > 0 and TRAIN_PERSISTENT_WORKERS,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(VAL_NUM_WORKERS_MIN, args.num_workers // VAL_NUM_WORKERS_DIVISOR),
        pin_memory=PIN_MEMORY,
        persistent_workers=VAL_PERSISTENT_WORKERS,
    )

    model = build_resnet18_classifier(pretrained=TRAIN_PRETRAINED).to(device=device)
    history = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history_path = args.output_dir / HISTORY_FILENAME
    history_path.write_text(data=json.dumps(history, indent=JSON_INDENT), encoding=JSON_ENCODING)
    plot_training_history(history=history, output_path=args.output_dir / TRAINING_CURVES_FILENAME)
    print(f"best epoch: {history['best_epoch']} | best val F1: {history['best_val_f1']:.4f}", flush=True)
    print(f"outputs written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
