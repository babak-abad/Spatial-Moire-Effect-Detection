"""Sanity checks for the config-driven pooled train / val / test split."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import (  # noqa: E402
    CLEAN_IMAGE_FOLDERS,
    DEFAULT_SEED,
    MOIRE_IMAGE_FOLDERS,
    SPLIT_MANIFEST_FILENAME,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
)
from spatial_moire.baseline.data import (  # noqa: E402
    list_folder_pairs,
    load_split_manifest,
    samples_from_pairs,
    save_split_manifest,
    split_train_val_test,
    validate_split_fractions,
)


def pooled_pair_groups() -> list[list[tuple[Path, Path]]]:
    return list_folder_pairs(clean_folders=CLEAN_IMAGE_FOLDERS, moire_folders=MOIRE_IMAGE_FOLDERS)


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


def sample_paths(samples: list[dict]) -> set[str]:
    return {sample["path"] for sample in samples}


def check_folder_pools_exist_and_pair_by_filename() -> None:
    assert len(CLEAN_IMAGE_FOLDERS) == len(MOIRE_IMAGE_FOLDERS), "clean/moire folder lists must be index-aligned"
    pair_groups = pooled_pair_groups()
    for clean_dir, moire_dir in zip(CLEAN_IMAGE_FOLDERS, MOIRE_IMAGE_FOLDERS, strict=True):
        assert clean_dir.is_dir() and moire_dir.is_dir(), f"configured folder pair does not exist: {clean_dir} / {moire_dir}"
    for pair_group in pair_groups:
        for moire_path, clean_path in pair_group:
            assert moire_path.name == clean_path.name, f"twins must share a filename: {moire_path} vs {clean_path}"
    total_pairs = sum(len(pair_group) for pair_group in pair_groups)
    print(f"ok: {len(CLEAN_IMAGE_FOLDERS)} folder pairs hold {total_pairs} filename-matched scene pairs")


def check_split_counts_match_fractions() -> None:
    total_pairs = sum(len(pair_group) for pair_group in pooled_pair_groups())
    train_samples, val_samples, test_samples = run_split()
    expected_train = 2 * round(total_pairs * TRAIN_FRACTION)
    expected_val = 2 * round(total_pairs * VAL_FRACTION)
    expected_test = 2 * total_pairs - expected_train - expected_val
    assert len(train_samples) == expected_train, f"train size {len(train_samples)} != {expected_train}"
    assert len(val_samples) == expected_val, f"val size {len(val_samples)} != {expected_val}"
    assert len(test_samples) == expected_test, f"test size {len(test_samples)} != {expected_test}"
    for name, samples in (("train", train_samples), ("val", val_samples), ("test", test_samples)):
        moire_count = sum(1 for sample in samples if sample["label"] == 1)
        clean_count = sum(1 for sample in samples if sample["label"] == 0)
        assert moire_count == clean_count == len(samples) // 2, f"{name} must stay class-balanced (one twin per pair)"
    print(f"ok: {total_pairs} pairs split into {expected_train}/{expected_val}/{expected_test} class-balanced samples")


def check_twins_never_separate() -> None:
    twins = {
        str(moire_path): str(clean_path)
        for pair_group in pooled_pair_groups()
        for moire_path, clean_path in pair_group
    }
    names = ("train", "val", "test")
    split_paths = [sample_paths(samples=samples) for samples in run_split()]
    for index, name in enumerate(names):
        for moire_path in split_paths[index]:
            clean_twin = twins.get(moire_path)
            if clean_twin is None:
                continue
            assert clean_twin in split_paths[index], f"the clean twin of a {name} moire image is missing from {name}"
            for other_index, other_paths in enumerate(split_paths):
                if other_index != index:
                    assert clean_twin not in other_paths, f"a {name} twin leaked into {names[other_index]}"
    print("ok: every moire/gt twin pair stays inside exactly one split")


def check_full_coverage_without_duplicates() -> None:
    all_samples = samples_from_pairs(pairs=[pair for pair_group in pooled_pair_groups() for pair in pair_group])
    combined = [sample for samples in run_split() for sample in samples]
    assert len(all_samples) == len(sample_paths(samples=all_samples)), "the pooled pools must not repeat a path"
    assert sample_paths(samples=combined) == sample_paths(samples=all_samples), "every pooled path must be covered"
    assert len(combined) == len(sample_paths(samples=combined)), "no path may appear in two splits"
    print(f"ok: the three splits cover all {len(all_samples)} pooled samples exactly once")


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
        pair_groups = list_folder_pairs(clean_folders=[clean_dir], moire_folders=[moire_dir])
        assert len(pair_groups) == 1 and len(pair_groups[0]) == 2
        empty_dir = root / "empty"
        empty_dir.mkdir()
        (clean_dir / "c.jpg").write_bytes(b"")
        attempts = (
            lambda: list_folder_pairs(clean_folders=[clean_dir], moire_folders=[moire_dir, moire_dir]),
            lambda: list_folder_pairs(clean_folders=[root / "missing_a"], moire_folders=[root / "missing_b"]),
            lambda: list_folder_pairs(clean_folders=[empty_dir], moire_folders=[moire_dir]),
            lambda: list_folder_pairs(clean_folders=[clean_dir], moire_folders=[moire_dir]),
        )
        for attempt in attempts:
            try:
                attempt()
            except ValueError:
                continue
            raise AssertionError("invalid folder pools must raise ValueError")
    print("ok: misaligned list lengths, missing/empty folders and filename mismatches raise ValueError")


def check_manifest_round_trip() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        clean_dir = root / "gt"
        moire_dir = root / "moire"
        clean_dir.mkdir()
        moire_dir.mkdir()
        for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
            (moire_dir / name).write_bytes(b"")
            (clean_dir / name).write_bytes(b"")
        train_samples, val_samples, test_samples = split_train_val_test(
            clean_folders=[clean_dir],
            moire_folders=[moire_dir],
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            seed=9,
        )
        assert (len(train_samples), len(val_samples), len(test_samples)) == (4, 2, 2)
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
        assert manifest["seed"] == 9
        assert (manifest["train_fraction"], manifest["val_fraction"], manifest["test_fraction"]) == (0.5, 0.25, 0.25)
        assert manifest["train"] == train_samples
        assert manifest["val"] == val_samples
        assert manifest["test"] == test_samples
        assert manifest["train_count"] == len(train_samples)
        assert manifest["val_count"] == len(val_samples)
        assert manifest["test_count"] == len(test_samples)
        replay_train, replay_val, replay_test = split_train_val_test(
            clean_folders=[clean_dir],
            moire_folders=[moire_dir],
            train_fraction=manifest["train_fraction"],
            val_fraction=manifest["val_fraction"],
            test_fraction=manifest["test_fraction"],
            seed=manifest["seed"],
        )
        assert (replay_train, replay_val, replay_test) == (train_samples, val_samples, test_samples), (
            "the split must be re-derivable from the recorded manifest settings (evaluate_baseline.py relies on this)"
        )
        assert load_split_manifest(path=root / "missing.json") is None, "a missing manifest must load as None"
    print("ok: split_manifest.json round-trips seed, fractions and all three sample lists")


def main() -> None:
    check_folder_pools_exist_and_pair_by_filename()
    check_split_counts_match_fractions()
    check_twins_never_separate()
    check_full_coverage_without_duplicates()
    check_determinism_follows_the_seed()
    check_invalid_fractions_are_rejected()
    check_folder_pools_are_validated()
    check_manifest_round_trip()
    print("all data split sanity checks passed")


if __name__ == "__main__":
    main()