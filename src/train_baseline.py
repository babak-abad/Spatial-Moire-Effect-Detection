"""Fine-tune an ImageNet-pretrained ResNet-18 as the deep-learning baseline for moire detection."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spatial_moire.baseline.augmentation import augmented_moire_dataset  # noqa: E402
from spatial_moire.baseline.data import (  # noqa: E402
    balanced_moire_dataset,
    balanced_shuffle_sampler,
    build_eval_transform,
    build_train_transform,
    moire_dataset,
    split_datasets,
)
from spatial_moire.baseline.engine import run_training  # noqa: E402
from spatial_moire.baseline.metrics import plot_training_history  # noqa: E402
from spatial_moire.baseline.model import build_resnet18_classifier  # noqa: E402
from config import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    GROUP_AUGMENT_COUNT_RANGE,
    HISTORY_FILENAME,
    JSON_ENCODING,
    JSON_INDENT,
    LEARNING_RATE,
    MOIRE_AUGMENT_PROB,
    MOIRE_LABEL,
    PIN_MEMORY,
    TRAIN_BALANCED_BATCHES,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_NUM_WORKERS,
    TRAIN_PERSISTENT_WORKERS,
    TRAIN_PRETRAINED,
    TRAIN_SHUFFLE,
    TRAINING_CURVES_FILENAME,
    VAL_NUM_WORKERS_DIVISOR,
    VAL_NUM_WORKERS_MIN,
    VAL_PERSISTENT_WORKERS,
    WEIGHT_DECAY,
)
from spatial_moire.moire_types import MOIRE_TYPES, type_probabilities  # noqa: E402
from spatial_moire.synthesis.groups import AUGMENTATION_GROUPS  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ResNet-18 moire detection baseline on the local LCDMoire subset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument(
        "--no-balance-batches",
        action="store_false",
        dest="balance_batches",
        default=TRAIN_BALANCED_BATCHES,
        help=(
            "disable exactly balanced train batches (batch_size // 2 clean + batch_size // 2 moire per batch) "
            "and fall back to plain shuffled batches"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the ultralytics-style batch progress bars and epoch summary rows",
    )
    parser.add_argument("--num-workers", type=int, default=TRAIN_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--moire-augment-prob",
        type=float,
        default=MOIRE_AUGMENT_PROB,
        help=(
            "probability that a moire-class train fetch receives one additional synthesized moire "
            "pattern; clean fetches never become moire and labels never change"
        ),
    )
    parser.add_argument(
        "--moire-augment-types",
        nargs="+",
        choices=sorted(MOIRE_TYPES),
        default=None,
    )
    parser.add_argument(
        "--group-augment",
        action="store_true",
        help=(
            "additionally corrupt every train image (moire-labeled and clean alike) with random "
            "capture-style augmentation groups; labels never change"
        ),
    )
    parser.add_argument(
        "--augment-groups",
        nargs="+",
        choices=sorted(AUGMENTATION_GROUPS),
        default=None,
        help="subset of augmentation groups to draw from (all groups by default)",
    )
    parser.add_argument(
        "--augment-group-count",
        nargs=2,
        type=int,
        metavar=("LOW", "HIGH"),
        default=None,
        help="inclusive range of augmentation groups applied per train image",
    )
    return parser.parse_args()


class Moire_Detection_DataLoader(DataLoader):
    """torch.utils.data.DataLoader subclass owning the moire-detection loader conventions.

    __init__ forwards every argument to DataLoader with the project defaults
    applied (pin_memory=PIN_MEMORY, batch_size and num_workers from config);
    the two role factories reproduce exactly the loader setup
    train_baseline.main used to build inline:

    for_training wires the train loop: a seeded torch.Generator for
    reproducible shuffling, an optional balanced_shuffle_sampler over the
    dataset's num_slots interleaved (clean, moire) index space — every
    even-length batch then holds exactly half clean and half moire items —
    shuffle=TRAIN_SHUFFLE only when no sampler is set, and persistent
    workers guarded by num_workers > 0. for_validation wires the val loop:
    sequential order (shuffle=False), num_workers scaled to
    max(VAL_NUM_WORKERS_MIN, num_workers // VAL_NUM_WORKERS_DIVISOR), and
    VAL_PERSISTENT_WORKERS. Loaders only move (image, label) batches; labels
    come from the dataset alone and never change here.
    """

    def __init__(
        self,
        dataset: moire_dataset,
        batch_size: int = TRAIN_BATCH_SIZE,
        shuffle: bool = False,
        sampler: Sampler[int] | None = None,
        num_workers: int = TRAIN_NUM_WORKERS,
        pin_memory: bool = PIN_MEMORY,
        persistent_workers: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            generator=generator,
        )

    @classmethod
    def for_training(
        cls,
        dataset: moire_dataset,
        batch_size: int,
        num_workers: int,
        seed: int,
        balance_batches: bool = False,
    ) -> Moire_Detection_DataLoader:
        if balance_batches and not hasattr(dataset, "num_slots"):
            raise ValueError(
                "balance_batches=True requires a dataset exposing num_slots "
                "(balanced_moire_dataset or augmented_moire_dataset with balanced=True)"
            )
        sampler = None
        if balance_batches:
            sampler = balanced_shuffle_sampler(num_slots=dataset.num_slots, seed=seed)
        return cls(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=TRAIN_SHUFFLE and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            persistent_workers=num_workers > 0 and TRAIN_PERSISTENT_WORKERS,
            generator=torch.Generator().manual_seed(seed),
        )

    @classmethod
    def for_validation(
        cls,
        dataset: moire_dataset,
        batch_size: int,
        num_workers: int,
    ) -> Moire_Detection_DataLoader:
        return cls(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(VAL_NUM_WORKERS_MIN, num_workers // VAL_NUM_WORKERS_DIVISOR),
            persistent_workers=VAL_PERSISTENT_WORKERS,
        )


def main() -> None:
    args = parse_args()
    seed_everything(seed=args.seed)
    if args.balance_batches and (args.batch_size < 2 or args.batch_size % 2 != 0):
        raise ValueError(f"balanced train batches require an even batch size of at least 2, got {args.batch_size}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(device=device)}", flush=True)

    train_samples, val_samples, test_samples = split_datasets()
    train_positives = sum(1 for sample in train_samples if sample["label"] == MOIRE_LABEL)
    val_positives = sum(1 for sample in val_samples if sample["label"] == MOIRE_LABEL)
    test_positives = sum(1 for sample in test_samples if sample["label"] == MOIRE_LABEL)
    print(
        f"train: {len(train_samples)} samples ({train_positives} moire / {len(train_samples) - train_positives} clean)",
        flush=True,
    )
    print(
        f"val: {len(val_samples)} samples ({val_positives} moire / {len(val_samples) - val_positives} clean)",
        flush=True,
    )
    print(
        f"test: {len(test_samples)} samples ({test_positives} moire / {len(test_samples) - test_positives} clean) held out by the split",
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_type_names = list(MOIRE_TYPES) if args.moire_augment_types is None else sorted(args.moire_augment_types)
    if args.moire_augment_prob > 0.0 or args.group_augment:
        train_group_names = list(AUGMENTATION_GROUPS) if args.augment_groups is None else sorted(args.augment_groups)
        group_count_range = (
            GROUP_AUGMENT_COUNT_RANGE if args.augment_group_count is None else tuple(args.augment_group_count)
        )
        train_set = augmented_moire_dataset(
            samples=train_samples,
            transform=build_train_transform(),
            moire_augment_prob=args.moire_augment_prob,
            type_names=train_type_names,
            seed=args.seed,
            group_augment=args.group_augment,
            group_names=train_group_names,
            num_groups_range=group_count_range,
            balanced=args.balance_batches,
        )
        if args.moire_augment_prob > 0.0:
            expected_moire_share = train_positives / len(train_samples)
            type_mix = ", ".join(
                f"{name} {probability:.0%}"
                for name, probability in type_probabilities(type_names=train_type_names).items()
            )
            if args.balance_batches:
                print(
                    f"moire augmentation: a {args.moire_augment_prob:.0%} share of the moire fetches in every balanced "
                    f"train batch receives one additional synthesized moire pattern via weighted moire type draw ({type_mix}); "
                    "the moire half of every batch always comes from real moire images",
                    flush=True,
                )
            else:
                print(
                    f"moire augmentation: a {args.moire_augment_prob:.0%} share of the {train_positives} moire train "
                    f"images receives one additional synthesized moire pattern via weighted moire type draw ({type_mix}); "
                    f"clean images never become moire (expected moire share stays {expected_moire_share:.0%})",
                    flush=True,
                )
        if args.group_augment:
            print(
                f"group augmentation: every train image (moire-labeled and clean alike) receives "
                f"{group_count_range[0]}-{group_count_range[1]} capture-style groups drawn from "
                f"{', '.join(train_group_names)}; labels never change",
                flush=True,
            )
    elif args.balance_batches:
        train_set = balanced_moire_dataset(samples=train_samples, transform=build_train_transform())
    else:
        train_set = moire_dataset(samples=train_samples, transform=build_train_transform())
    val_set = moire_dataset(samples=val_samples, transform=build_eval_transform())
    if args.balance_batches:
        print(
            f"balanced batches: every train batch of {args.batch_size} holds {args.batch_size // 2} clean + "
            f"{args.batch_size // 2} moire items ({train_set.num_slots} slots per class per epoch)",
            flush=True,
        )
    train_loader = Moire_Detection_DataLoader.for_training(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        balance_batches=args.balance_batches,
    )
    val_loader = Moire_Detection_DataLoader.for_validation(
        dataset=val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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
        progress=not args.no_progress,
    )

    history_path = args.output_dir / HISTORY_FILENAME
    history_path.write_text(data=json.dumps(history, indent=JSON_INDENT), encoding=JSON_ENCODING)
    plot_training_history(history=history, output_path=args.output_dir / TRAINING_CURVES_FILENAME)
    print(f"best epoch: {history['best_epoch']} | best val F1: {history['best_val_f1']:.4f}", flush=True)
    print(f"outputs written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
