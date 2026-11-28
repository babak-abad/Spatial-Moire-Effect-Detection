"""Sanity checks for the folder-list train / val / test split (split_datasets)."""

from __future__ import annotations

import sys
import tempfile
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import (  # noqa: E402
    CLEAN_IMAGE_FOLDERS,
    CLEAN_IMAGE_FOLDERS_TEST,
    CLEAN_IMAGE_FOLDERS_TRAIN,
    CLEAN_IMAGE_FOLDERS_VAL,
    CLEAN_LABEL,
    MOIRE_IMAGE_FOLDERS,
    MOIRE_IMAGE_FOLDERS_TEST,
    MOIRE_IMAGE_FOLDERS_TRAIN,
    MOIRE_IMAGE_FOLDERS_VAL,
    MOIRE_LABEL,
)
from spatial_moire.baseline.data import (  # noqa: E402
    collect_class_samples,
    split_datasets,
)


def run_split() -> tuple[list[dict], list[dict], list[dict]]:
    return split_datasets()


@lru_cache(maxsize=None)
def collect_pool(folders: tuple[Path, ...], label: int) -> tuple[dict, ...]:
    return tuple(collect_class_samples(folders=list(folders), label=label))


def pool_samples() -> list[dict]:
    return list(collect_pool(folders=tuple(CLEAN_IMAGE_FOLDERS), label=CLEAN_LABEL)) + list(
        collect_pool(folders=tuple(MOIRE_IMAGE_FOLDERS), label=MOIRE_LABEL)
    )


def sample_paths(samples: list[dict]) -> set[str]:
    return {sample["path"] for sample in samples}


def check_pools_are_split_unions() -> None:
    assert CLEAN_IMAGE_FOLDERS == CLEAN_IMAGE_FOLDERS_TRAIN + CLEAN_IMAGE_FOLDERS_VAL + CLEAN_IMAGE_FOLDERS_TEST
    assert MOIRE_IMAGE_FOLDERS == MOIRE_IMAGE_FOLDERS_TRAIN + MOIRE_IMAGE_FOLDERS_VAL + MOIRE_IMAGE_FOLDERS_TEST
    assert len(CLEAN_IMAGE_FOLDERS) == len(set(CLEAN_IMAGE_FOLDERS)), "no clean folder may be listed twice"
    assert len(MOIRE_IMAGE_FOLDERS) == len(set(MOIRE_IMAGE_FOLDERS)), "no moire folder may be listed twice"
    print(
        f"ok: pools are the ordered unions of the six lists "
        f"({len(CLEAN_IMAGE_FOLDERS_TRAIN)}+{len(CLEAN_IMAGE_FOLDERS_VAL)}+{len(CLEAN_IMAGE_FOLDERS_TEST)} clean, "
        f"{len(MOIRE_IMAGE_FOLDERS_TRAIN)}+{len(MOIRE_IMAGE_FOLDERS_VAL)}+{len(MOIRE_IMAGE_FOLDERS_TEST)} moire folders)"
    )


def check_lists_non_empty_and_disjoint() -> None:
    for class_name, class_lists in (
        ("clean", (CLEAN_IMAGE_FOLDERS_TRAIN, CLEAN_IMAGE_FOLDERS_VAL, CLEAN_IMAGE_FOLDERS_TEST)),
        ("moire", (MOIRE_IMAGE_FOLDERS_TRAIN, MOIRE_IMAGE_FOLDERS_VAL, MOIRE_IMAGE_FOLDERS_TEST)),
    ):
        for split_name, folders in zip(("train", "val", "test"), class_lists, strict=True):
            assert folders, f"the {class_name} {split_name} folder list must not be empty"
        seen: set[Path] = set()
        for folders in class_lists:
            for folder in folders:
                assert folder not in seen, f"{class_name} folder {folder} is listed for more than one split"
                seen.add(folder)
    print("ok: every list is non-empty and no source folder feeds two splits of one class")


def check_folder_pools_hold_images() -> None:
    clean_pool = collect_pool(folders=tuple(CLEAN_IMAGE_FOLDERS), label=CLEAN_LABEL)
    moire_pool = collect_pool(folders=tuple(MOIRE_IMAGE_FOLDERS), label=MOIRE_LABEL)
    assert all(sample["label"] == CLEAN_LABEL for sample in clean_pool), "the clean pool must label every sample 0"
    assert all(sample["label"] == MOIRE_LABEL for sample in moire_pool), "the moire pool must label every sample 1"
    print(
        f"ok: {len(CLEAN_IMAGE_FOLDERS)} clean folders hold {len(clean_pool)} images, "
        f"{len(MOIRE_IMAGE_FOLDERS)} moire folders hold {len(moire_pool)} images"
    )


def check_splits_partition_the_pools(
    train_samples: list[dict], val_samples: list[dict], test_samples: list[dict]
) -> None:
    all_samples = pool_samples()
    combined = train_samples + val_samples + test_samples
    assert len(sample_paths(samples=combined)) == len(combined), "no path may appear in two splits"
    assert sample_paths(samples=combined) == sample_paths(samples=all_samples), "every pooled path must be covered"
    label_by_path = {sample["path"]: sample["label"] for sample in all_samples}
    assert all(sample["label"] == label_by_path[sample["path"]] for sample in combined), "labels must follow the pools"
    print(f"ok: the three splits cover all {len(all_samples)} pooled samples exactly once")


def check_splits_follow_their_folder_lists(
    train_samples: list[dict], val_samples: list[dict], test_samples: list[dict]
) -> None:
    for name, samples, clean_folders, moire_folders in (
        ("train", train_samples, CLEAN_IMAGE_FOLDERS_TRAIN, MOIRE_IMAGE_FOLDERS_TRAIN),
        ("val", val_samples, CLEAN_IMAGE_FOLDERS_VAL, MOIRE_IMAGE_FOLDERS_VAL),
        ("test", test_samples, CLEAN_IMAGE_FOLDERS_TEST, MOIRE_IMAGE_FOLDERS_TEST),
    ):
        expected = list(collect_pool(folders=tuple(clean_folders), label=CLEAN_LABEL)) + list(
            collect_pool(folders=tuple(moire_folders), label=MOIRE_LABEL)
        )
        assert samples == expected, f"the {name} split must equal its folder lists' samples in (folder, filename) order"
    print("ok: every split equals its two folder lists' samples in (folder, filename) order")


def check_per_split_class_balance(
    train_samples: list[dict], val_samples: list[dict], test_samples: list[dict]
) -> None:
    for name, samples in zip(("train", "val", "test"), (train_samples, val_samples, test_samples), strict=True):
        clean_count = sum(1 for sample in samples if sample["label"] == CLEAN_LABEL)
        moire_count = len(samples) - clean_count
        assert clean_count > 0, f"the {name} split must hold clean samples"
        assert moire_count > 0, f"the {name} split must hold moire samples"
        print(f"ok: {name} holds {clean_count} clean and {moire_count} moire samples")


def check_determinism() -> None:
    run_a = run_split()
    assert run_split() == run_a, "repeated calls must reproduce the exact same splits"
    print("ok: repeated calls reproduce the splits exactly")


def check_invalid_folder_lists_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        clean_dir = root / "gt"
        moire_dir = root / "moire"
        empty_dir = root / "empty"
        clean_dir.mkdir()
        moire_dir.mkdir()
        empty_dir.mkdir()
        (clean_dir / "a.jpg").write_bytes(b"")
        (moire_dir / "b.jpg").write_bytes(b"")
        attempts = (
            ("an empty clean train list", lambda: split_datasets(clean_train_folders=[])),
            ("a clean folder listed for train and val", lambda: split_datasets(
                clean_train_folders=[clean_dir],
                clean_val_folders=[clean_dir],
            )),
            ("a missing moire train folder", lambda: split_datasets(moire_train_folders=[root / "missing"])),
            ("an empty moire val folder", lambda: split_datasets(moire_val_folders=[empty_dir])),
        )
        for description, attempt in attempts:
            try:
                attempt()
            except ValueError:
                continue
            raise AssertionError(f"{description} must raise ValueError")
        print("ok: empty lists, double-listed folders, missing folders and empty folders raise ValueError")


def main() -> None:
    check_pools_are_split_unions()
    check_lists_non_empty_and_disjoint()
    check_folder_pools_hold_images()
    train_samples, val_samples, test_samples = run_split()
    check_splits_partition_the_pools(
        train_samples=train_samples, val_samples=val_samples, test_samples=test_samples
    )
    check_splits_follow_their_folder_lists(
        train_samples=train_samples, val_samples=val_samples, test_samples=test_samples
    )
    check_per_split_class_balance(
        train_samples=train_samples, val_samples=val_samples, test_samples=test_samples
    )
    check_determinism()
    check_invalid_folder_lists_are_rejected()
    print("all data split sanity checks passed")


if __name__ == "__main__":
    main()