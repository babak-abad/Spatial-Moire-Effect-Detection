"""Sanity checks for the per-class unit-based train / val / test split."""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import (  # noqa: E402
    CLEAN_IMAGE_FOLDERS,
    CLEAN_LABEL,
    DEFAULT_SEED,
    FOLDER_UNIT_MAX_IMAGES,
    MOIRE_IMAGE_FOLDERS,
    MOIRE_LABEL,
    SPLIT_MANIFEST_FILENAME,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
)
from spatial_moire.baseline.data import (  # noqa: E402
    SPLIT_MANIFEST_ALGORITHM,
    build_split_units,
    collect_class_samples,
    load_split_manifest,
    save_split_manifest,
    split_train_val_test,
    validate_split_fractions,
)


def run_split(seed: int = DEFAULT_SEED) -> tuple[list[dict], list[dict], list[dict]]:
    return split_train_val_test(clean_folders=CLEAN_IMAGE_FOLDERS, moire_folders=MOIRE_IMAGE_FOLDERS, seed=seed)


def run_split_with_fractions(train_fraction: float, val_fraction: float, test_fraction: float) -> list[dict]:
    train_samples, _val_samples, _test_samples = split_train_val_test(
        clean_folders=CLEAN_IMAGE_FOLDERS,
        moire_folders=MOIRE_IMAGE_FOLDERS,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    return train_samples


def stream_units(label: int) -> list[list[dict]]:
    folders = CLEAN_IMAGE_FOLDERS if label == CLEAN_LABEL else MOIRE_IMAGE_FOLDERS
    return build_split_units(folders=folders, label=label)


def shuffled_stream_units(label: int, seed: int = DEFAULT_SEED) -> list[list[dict]]:
    units = stream_units(label=label)
    random.Random(seed + (0 if label == CLEAN_LABEL else 1)).shuffle(units)
    return units


def sample_paths(samples: list[dict]) -> set[str]:
    return {sample["path"] for sample in samples}


def check_folder_pools_exist_and_hold_images() -> None:
    clean_pool = collect_class_samples(folders=CLEAN_IMAGE_FOLDERS, label=CLEAN_LABEL)
    moire_pool = collect_class_samples(folders=MOIRE_IMAGE_FOLDERS, label=MOIRE_LABEL)
    assert all(sample["label"] == CLEAN_LABEL for sample in clean_pool), "the clean pool must label every sample 0"
    assert all(sample["label"] == MOIRE_LABEL for sample in moire_pool), "the moire pool must label every sample 1"
    print(
        f"ok: {len(CLEAN_IMAGE_FOLDERS)} clean pool folders hold {len(clean_pool)} images, "
        f"{len(MOIRE_IMAGE_FOLDERS)} moire pool folders hold {len(moire_pool)} images"
    )


def check_split_counts_match_fractions() -> None:
    train_samples, val_samples, test_samples = run_split()
    for label in (CLEAN_LABEL, MOIRE_LABEL):
        units = shuffled_stream_units(label=label)
        train_count = round(len(units) * TRAIN_FRACTION)
        val_count = round(len(units) * VAL_FRACTION)
        slices = (
            ("train", train_samples, units[:train_count]),
            ("val", val_samples, units[train_count : train_count + val_count]),
            ("test", test_samples, units[train_count + val_count :]),
        )
        for name, samples, unit_slice in slices:
            expected = sum(len(unit) for unit in unit_slice)
            actual = sum(1 for sample in samples if sample["label"] == label)
            assert actual == expected, f"{name} holds {actual} label-{label} samples, expected {expected}"
    total_samples = sum(len(unit) for label in (CLEAN_LABEL, MOIRE_LABEL) for unit in stream_units(label=label))
    print(f"ok: {total_samples} pooled samples split per stream with rounded unit cuts, test gets the remainder")


def check_units_never_straddle_splits() -> None:
    split_paths = [sample_paths(samples=samples) for samples in run_split()]
    for label in (CLEAN_LABEL, MOIRE_LABEL):
        for unit in stream_units(label=label):
            unit_paths = {sample["path"] for sample in unit}
            homes = [index for index, paths in enumerate(split_paths) if unit_paths & paths]
            assert len(homes) == 1, (
                f"a {len(unit)}-sample label-{label} unit must stay inside exactly one split, straddled {homes}"
            )
    print("ok: atomic folder units and merged same-stem units never straddle a split boundary")


def check_full_coverage_without_duplicates() -> None:
    all_samples = collect_class_samples(folders=CLEAN_IMAGE_FOLDERS, label=CLEAN_LABEL)
    all_samples += collect_class_samples(folders=MOIRE_IMAGE_FOLDERS, label=MOIRE_LABEL)
    combined = [sample for samples in run_split() for sample in samples]
    assert len(all_samples) == len(sample_paths(samples=all_samples)), "the class pools must not repeat a path"
    assert sample_paths(samples=combined) == sample_paths(samples=all_samples), "every pooled path must be covered"
    assert len(combined) == len(sample_paths(samples=combined)), "no path may appear in two splits"
    print(f"ok: the three splits cover all {len(all_samples)} pooled samples exactly once")


def check_split_lists_order_clean_before_moire() -> None:
    for name, samples in zip(("train", "val", "test"), run_split(), strict=True):
        labels = [sample["label"] for sample in samples]
        clean_count = labels.count(CLEAN_LABEL)
        assert labels == [CLEAN_LABEL] * clean_count + [MOIRE_LABEL] * (len(labels) - clean_count), (
            f"the {name} list must hold all clean samples before all moire samples"
        )
    print("ok: every split list orders its clean samples before its moire samples")


def check_stream_shuffles_follow_seed_offsets() -> None:
    train_samples, _val_samples, _test_samples = run_split()
    for label in (CLEAN_LABEL, MOIRE_LABEL):
        units = shuffled_stream_units(label=label)
        train_count = round(len(units) * TRAIN_FRACTION)
        expected = [sample["path"] for unit in units[:train_count] for sample in unit]
        actual = [sample["path"] for sample in train_samples if sample["label"] == label]
        stream = "clean" if label == CLEAN_LABEL else "moire"
        assert actual == expected, f"the {stream} train stream must follow its dedicated Random(seed offset) shuffle"
    print("ok: the clean stream shuffles with Random(seed), the moire stream with Random(seed + 1)")


def check_determinism_follows_the_seed() -> None:
    run_a = run_split()
    assert run_split() == run_a, "identical seeds must reproduce the exact same split"
    assert run_split(seed=DEFAULT_SEED + 1) != run_a, "different seeds must draw a different split"
    print("ok: identical seeds reproduce the split, different seeds change it")


def check_invalid_fractions_are_rejected() -> None:
    attempts = (
        lambda: run_split_with_fractions(train_fraction=0.5, val_fraction=0.2, test_fraction=0.2),
        lambda: run_split_with_fractions(train_fraction=1.1, val_fraction=0.0, test_fraction=-0.1),
    )
    for attempt in attempts:
        try:
            attempt()
        except ValueError:
            continue
        raise AssertionError("invalid split fractions must raise ValueError")
    validate_split_fractions(train_fraction=0.4, val_fraction=0.4, test_fraction=0.2)
    print("ok: fractions outside [0, 1] or not summing to 1 raise ValueError")


def check_folder_pools_are_validated() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        clean_dir = root / "gt"
        moire_dir = root / "moire"
        clean_dir.mkdir()
        moire_dir.mkdir()
        for name in ("a.jpg", "b.jpg"):
            (moire_dir / name).write_bytes(b"")
            (clean_dir / name).write_bytes(b"")
        empty_dir = root / "empty"
        empty_dir.mkdir()
        attempts = (
            lambda: collect_class_samples(folders=[root / "missing_a"], label=CLEAN_LABEL),
            lambda: collect_class_samples(folders=[empty_dir], label=CLEAN_LABEL),
            lambda: build_split_units(folders=[root / "missing_b"], label=MOIRE_LABEL),
        )
        for attempt in attempts:
            try:
                attempt()
            except ValueError:
                continue
            raise AssertionError("missing or empty folders must raise ValueError")
        extra_moire = root / "extra_moire"
        extra_moire.mkdir()
        (extra_moire / "c.jpg").write_bytes(b"")
        train_samples, _val_samples, _test_samples = split_train_val_test(
            clean_folders=[clean_dir],
            moire_folders=[moire_dir, extra_moire],
            train_fraction=1.0,
            val_fraction=0.0,
            test_fraction=0.0,
        )
        labels = [sample["label"] for sample in train_samples]
        assert labels == [CLEAN_LABEL, CLEAN_LABEL, MOIRE_LABEL, MOIRE_LABEL, MOIRE_LABEL], (
            "pools of unequal size and unaligned folder lists must split without any pairing requirement"
        )
    print("ok: missing/empty folders raise ValueError; pools of unequal size split without pairing")


def check_manifest_round_trip() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        clean_dirs = [root / f"gt_{index}" for index in range(8)]
        moire_dirs = [root / f"moire_{index}" for index in range(8)]
        for index, (clean_dir, moire_dir) in enumerate(zip(clean_dirs, moire_dirs, strict=True)):
            clean_dir.mkdir()
            moire_dir.mkdir()
            (moire_dir / f"{index}.jpg").write_bytes(b"")
            (clean_dir / f"{index}.jpg").write_bytes(b"")
        train_samples, val_samples, test_samples = split_train_val_test(
            clean_folders=clean_dirs,
            moire_folders=moire_dirs,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            seed=9,
        )
        assert (len(train_samples), len(val_samples), len(test_samples)) == (8, 4, 4)
        manifest_path = root / SPLIT_MANIFEST_FILENAME
        save_split_manifest(
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            path=manifest_path,
            seed=9,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
        )
        manifest = load_split_manifest(path=manifest_path)
        assert manifest is not None, "load_split_manifest must read the saved manifest"
        assert manifest["algorithm"] == SPLIT_MANIFEST_ALGORITHM, "the manifest must pin the split algorithm version"
        assert manifest["folder_unit_max_images"] == FOLDER_UNIT_MAX_IMAGES, (
            "the manifest must record the unit-grouping threshold it was built with"
        )
        assert manifest["seed"] == 9
        assert (manifest["train_fraction"], manifest["val_fraction"], manifest["test_fraction"]) == (0.5, 0.25, 0.25)
        assert manifest["train"] == train_samples
        assert manifest["val"] == val_samples
        assert manifest["test"] == test_samples
        assert manifest["train_count"] == len(train_samples)
        assert manifest["val_count"] == len(val_samples)
        assert manifest["test_count"] == len(test_samples)
        replay_train, replay_val, replay_test = split_train_val_test(
            clean_folders=clean_dirs,
            moire_folders=moire_dirs,
            train_fraction=manifest["train_fraction"],
            val_fraction=manifest["val_fraction"],
            test_fraction=manifest["test_fraction"],
            seed=manifest["seed"],
        )
        assert (replay_train, replay_val, replay_test) == (train_samples, val_samples, test_samples), (
            "the split must be re-derivable from the recorded manifest settings (evaluate_baseline.py relies on this)"
        )
        assert load_split_manifest(path=root / "missing.json") is None, "a missing manifest must load as None"
    print("ok: split_manifest.json round-trips algorithm, seed, fractions and all three sample lists")


def main() -> None:
    check_folder_pools_exist_and_hold_images()
    check_split_counts_match_fractions()
    check_units_never_straddle_splits()
    check_full_coverage_without_duplicates()
    check_split_lists_order_clean_before_moire()
    check_stream_shuffles_follow_seed_offsets()
    check_determinism_follows_the_seed()
    check_invalid_fractions_are_rejected()
    check_folder_pools_are_validated()
    check_manifest_round_trip()
    print("all data split sanity checks passed")


if __name__ == "__main__":
    main()