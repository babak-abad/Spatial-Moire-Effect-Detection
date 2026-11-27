"""Save the training augmentation pipeline's outputs for clean and moire inputs as an HTML preview.

Runs the exact augmented_moire_dataset used by train_baseline.py (with
transform=None so the returned PIL images stay viewable) over a few paired
clean / real-moire scenes drawn from the first configured clean/moire folder
pair for several passes, captures the drawn recipe
with delegate-only recorders around the training dispatch points, saves every
input and output JPEG plus thumbnails, and writes a self-contained HTML
gallery and a JSON manifest into the output folder for offline analysis.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import (  # noqa: E402
    AUGMENTATION_PREVIEW_HTML_FILENAME,
    AUGMENTATION_PREVIEW_INPUT_FILENAME_TEMPLATE,
    AUGMENTATION_PREVIEW_MANIFEST_FILENAME,
    AUGMENTATION_PREVIEW_NUM_SCENES,
    AUGMENTATION_PREVIEW_OUTPUT_DIR,
    AUGMENTATION_PREVIEW_OUTPUT_FILENAME_TEMPLATE,
    AUGMENTATION_PREVIEW_PASSES,
    AUGMENTATION_PREVIEW_THUMB_JPEG_QUALITY,
    AUGMENTATION_PREVIEW_THUMB_PREFIX,
    AUGMENTATION_PREVIEW_THUMB_SIZE,
    CLEAN_IMAGE_FOLDERS,
    CLEAN_LABEL,
    DEFAULT_SEED,
    GROUP_AUGMENT_COUNT_RANGE,
    JPEG_QUALITY,
    JSON_ENCODING,
    JSON_INDENT,
    MOIRE_AUGMENT_PROB,
    MOIRE_IMAGE_FOLDERS,
    MOIRE_LABEL,
)
from spatial_moire.baseline.augmentation import augmented_moire_dataset  # noqa: E402
import spatial_moire.baseline.augmentation as augmentation_module  # noqa: E402
from spatial_moire.baseline.data import list_folder_pairs, samples_from_pairs  # noqa: E402
from spatial_moire.moire_types import MOIRE_TYPES, type_probabilities  # noqa: E402
from spatial_moire.synthesis.groups import AUGMENTATION_GROUPS  # noqa: E402

PAGE_TITLE: str = "Train augmentation preview — Spatial-Moire-Effect-Detection"
PATH_SHORT_TEXT: dict[str, str] = {
    "clean_to_moire": "clean → moire (synthesized)",
    "clean_groups_only": "clean, groups only",
    "moire_groups_only": "real moire, groups only",
    "untouched": "untouched",
}

PREVIEW_CSS: str = """
body { font-family: system-ui, "Segoe UI", sans-serif; margin: 24px auto; max-width: 1400px; padding: 0 16px; color: #1b1b1b; background: #fafafa; }
h1 { font-size: 20px; margin-bottom: 4px; }
h2 { font-size: 15px; margin: 28px 0 8px; border-bottom: 2px solid #2e7d32; padding-bottom: 4px; }
table { border-collapse: collapse; font-size: 12px; margin: 8px 0; }
th, td { border: 1px solid #ccc; padding: 3px 8px; text-align: left; }
th { background: #eeeeee; }
.tables { display: flex; gap: 24px; flex-wrap: wrap; }
.row { display: grid; gap: 10px; margin: 14px 0; align-items: start; }
.row-head { font-size: 12px; font-weight: bold; color: #444; }
.cell img { width: 100%; height: auto; display: block; border: 1px solid #ddd; background: #ffffff; }
.caption { font-size: 11px; margin-top: 3px; color: #333; line-height: 1.35; }
.badge { display: inline-block; padding: 1px 7px; border-radius: 3px; font-size: 11px; color: #ffffff; }
.badge-moire { background: #c62828; }
.badge-clean { background: #2e7d32; }
.notes { font-size: 12px; color: #555; max-width: 1100px; }
.note { font-size: 11px; color: #777; }
footer { font-size: 12px; color: #555; margin-top: 36px; border-top: 1px solid #ccc; padding-top: 10px; }
code { background: #eeeeee; padding: 1px 5px; border-radius: 3px; font-size: 11px; word-break: break-all; }
"""


class draw_recorder:
    """Install delegate-only recorders around the training dispatch points.

    The wrapped names are exactly the ones augmented_moire_dataset dispatches
    through (sample_moire_type, MOIRE_GENERATORS, apply_group_augmentation).
    Every wrapper calls the original function with the original rng and
    returns its result unchanged, so pixels and rng draws stay identical to
    training; the wrappers only append what was drawn to records[-1]. The
    module names are restored on exit.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self._originals: tuple[Any, Any, Any] | None = None

    def __enter__(self) -> "draw_recorder":
        original_sample_moire_type = augmentation_module.sample_moire_type
        original_generators = augmentation_module.MOIRE_GENERATORS
        original_apply_group_augmentation = augmentation_module.apply_group_augmentation
        self._originals = (
            original_sample_moire_type,
            original_generators,
            original_apply_group_augmentation,
        )
        records = self._records

        def sample_moire_type_recorder(rng: np.random.Generator, type_names: list[str] | None = None) -> str:
            type_name = original_sample_moire_type(rng=rng, type_names=type_names)
            if records:
                records[-1]["moire_type"] = type_name
            return type_name

        class recording_generator_dict(dict):
            def __getitem__(self, generator_name: Any) -> Any:
                generator = original_generators[generator_name]

                def generator_recorder(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
                    result, params = generator(image=image, rng=rng)
                    if records:
                        records[-1]["generator"] = str(generator_name)
                        records[-1]["generator_params"] = params
                    return result, params

                return generator_recorder

        def apply_group_augmentation_recorder(
            image: np.ndarray,
            rng: np.random.Generator,
            group_subset: list[str] | None = None,
            num_groups_range: tuple[int, int] = GROUP_AUGMENT_COUNT_RANGE,
        ) -> tuple[np.ndarray, dict]:
            result, params = original_apply_group_augmentation(
                image=image,
                rng=rng,
                group_subset=group_subset,
                num_groups_range=num_groups_range,
            )
            if records:
                records[-1]["groups"] = params
            return result, params

        augmentation_module.sample_moire_type = sample_moire_type_recorder
        augmentation_module.MOIRE_GENERATORS = recording_generator_dict(original_generators)
        augmentation_module.apply_group_augmentation = apply_group_augmentation_recorder
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._originals is None:
            return
        original_sample_moire_type, original_generators, original_apply_group_augmentation = self._originals
        augmentation_module.sample_moire_type = original_sample_moire_type
        augmentation_module.MOIRE_GENERATORS = original_generators
        augmentation_module.apply_group_augmentation = original_apply_group_augmentation
        self._originals = None


def select_scene_pairs(num_scenes: int, rng: np.random.Generator) -> list[tuple[Path, Path]]:
    """Pick num_scenes paired (moire, clean) scenes from the first config folder pair with a seeded draw, kept in filename order.

    Only the first clean/moire folder pair is used so every scene stem stays
    unique: the gallery and the manifest key records by (kind, stem) and stems
    repeat across folder pairs.
    """
    if num_scenes < 1:
        raise SystemExit("--num-scenes must be at least 1")
    pairs = list_folder_pairs(clean_folders=CLEAN_IMAGE_FOLDERS, moire_folders=MOIRE_IMAGE_FOLDERS)[0]
    if num_scenes > len(pairs):
        raise SystemExit(
            f"--num-scenes {num_scenes} exceeds the {len(pairs)} paired scenes under {MOIRE_IMAGE_FOLDERS[0]}"
        )
    chosen = sorted(int(index) for index in rng.permutation(len(pairs))[:num_scenes])
    return [pairs[index] for index in chosen]


def thumbnail_filename(image_filename: str) -> str:
    stem = Path(image_filename).stem
    return f"{AUGMENTATION_PREVIEW_THUMB_PREFIX}{stem}.jpg"


def save_thumbnail(image: Image.Image, path: Path) -> None:
    thumbnail = image.copy()
    thumbnail.thumbnail(size=AUGMENTATION_PREVIEW_THUMB_SIZE)
    thumbnail.save(fp=path, quality=AUGMENTATION_PREVIEW_THUMB_JPEG_QUALITY)


def classify_path(source_kind: str, record: dict[str, Any]) -> str:
    """Name the __getitem__ path taken, from what the recorders captured."""
    if "moire_type" in record:
        return "clean_to_moire"
    if "groups" in record:
        return f"{source_kind}_groups_only"
    return "untouched"


def copy_inputs(
    samples: list[dict[str, Any]],
    scene_kinds: list[str],
    scene_stems: list[str],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Copy the source images into the output folder (plus thumbnails) so the HTML is portable."""
    inputs: list[dict[str, Any]] = []
    for sample, kind, stem in zip(samples, scene_kinds, scene_stems, strict=True):
        source = Path(sample["path"])
        input_name = AUGMENTATION_PREVIEW_INPUT_FILENAME_TEMPLATE.format(kind=kind, stem=stem, suffix=source.suffix)
        shutil.copyfile(src=source, dst=output_dir / input_name)
        with Image.open(fp=output_dir / input_name) as image:
            save_thumbnail(image=image, path=output_dir / thumbnail_filename(input_name))
        inputs.append(
            {
                "kind": kind,
                "stem": stem,
                "label": int(sample["label"]),
                "source": str(source),
                "file": input_name,
            }
        )
        print(f"wrote {input_name}", flush=True)
    return inputs


def generate_outputs(
    dataset: augmented_moire_dataset,
    samples: list[dict[str, Any]],
    scene_kinds: list[str],
    scene_stems: list[str],
    output_dir: Path,
    passes: int,
    jpeg_quality: int,
) -> tuple[list[dict[str, Any]], list[tuple[int, np.ndarray]]]:
    """Fetch every pass with the recorders installed and save each returned image.

    One dataset instance serves all passes and its rng advances across them,
    like epochs in training. Returns (records, pass-1 label/array pairs) where
    the pass-1 pairs feed the pixel-neutrality replay check.
    """
    records: list[dict[str, Any]] = []
    pass_one: list[tuple[int, np.ndarray]] = []
    with draw_recorder(records=records):
        for pass_num in range(1, passes + 1):
            for index, sample in enumerate(samples):
                records.append(
                    {
                        "pass": pass_num,
                        "sample_index": index,
                        "source_kind": scene_kinds[index],
                        "source_stem": scene_stems[index],
                        "source": sample["path"],
                        "source_label": int(sample["label"]),
                    }
                )
                image, label = dataset[index]
                record = records[-1]
                record["returned_label"] = int(label)
                record["path"] = classify_path(source_kind=scene_kinds[index], record=record)
                output_name = AUGMENTATION_PREVIEW_OUTPUT_FILENAME_TEMPLATE.format(
                    kind=scene_kinds[index],
                    stem=scene_stems[index],
                    pass_num=pass_num,
                )
                image.save(fp=output_dir / output_name, quality=jpeg_quality)
                save_thumbnail(image=image, path=output_dir / thumbnail_filename(output_name))
                record["file"] = output_name
                if pass_num == 1:
                    pass_one.append((int(label), np.asarray(image, dtype=np.uint8).copy()))
                print(f"wrote {output_name}", flush=True)
    return records, pass_one


def verify_recording_is_pixel_neutral(
    dataset_kwargs: dict[str, Any],
    pass_one: list[tuple[int, np.ndarray]],
) -> None:
    """Rebuild the dataset with the same seed, without recorders, and compare pass 1."""
    replay_set = augmented_moire_dataset(**dataset_kwargs)
    for index, (expected_label, expected_array) in enumerate(pass_one):
        image, label = replay_set[index]
        assert int(label) == expected_label, f"replay label mismatch at sample {index}"
        assert np.array_equal(np.asarray(image, dtype=np.uint8), expected_array), (
            f"replay pixels differ at sample {index}: the recording wrappers must be pixel-neutral"
        )
    print(f"ok: recorder-free pass-1 replay matches all {len(pass_one)} outputs pixel-for-pixel", flush=True)


def compute_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the drawn recipes: path counts, type distribution, group frequencies."""
    clean_records = [record for record in records if record["source_kind"] == "clean"]
    synthesized_records = [record for record in records if "moire_type" in record]
    grouped_records = [record for record in records if "groups" in record]
    num_groups_values = [int(record["groups"]["num_groups"]) for record in grouped_records]
    return {
        "num_records": len(records),
        "path_counts": dict(Counter(record["path"] for record in records)),
        "num_clean_records": len(clean_records),
        "num_converted_clean": sum(1 for record in clean_records if record["path"] == "clean_to_moire"),
        "num_synthesized": len(synthesized_records),
        "type_counts": dict(Counter(record["moire_type"] for record in synthesized_records)),
        "num_grouped_draws": len(grouped_records),
        "group_counts": dict(
            Counter(entry["group"] for record in grouped_records for entry in record["groups"]["groups"])
        ),
        "mean_groups_per_draw": (sum(num_groups_values) / len(num_groups_values)) if num_groups_values else 0.0,
    }


def label_badge(label: int) -> str:
    if label == MOIRE_LABEL:
        return '<span class="badge badge-moire">label 1 · moire</span>'
    return '<span class="badge badge-clean">label 0 · clean</span>'


def recipe_html(record: dict[str, Any]) -> str:
    """One-line recipe of what was drawn for this fetch (type · generator, groups)."""
    lines: list[str] = []
    if "moire_type" in record:
        lines.append(f"{escape(record['moire_type'])} · {escape(record['generator'])}")
    if "groups" in record:
        applied = [entry["group"] for entry in record["groups"]["groups"]]
        lines.append("groups: " + (escape(" · ".join(applied)) if applied else "none drawn"))
    if not lines:
        lines.append("no augmentation")
    return "<br>".join(lines)


def render_scene_section(
    kind: str,
    section_title: str,
    inputs: list[dict[str, Any]],
    records_by_key: dict[tuple[str, str, int], dict[str, Any]],
    passes: int,
) -> str:
    """One gallery section: per scene an input cell plus one output cell per pass."""
    grid_style = f'style="grid-template-columns: repeat({passes + 1}, 1fr)"'
    header_cells = "<div>input</div>" + "".join(f"<div>pass {pass_num}</div>" for pass_num in range(1, passes + 1))
    rows = [f'<div class="row row-head" {grid_style}>{header_cells}</div>']
    for source_input in (entry for entry in inputs if entry["kind"] == kind):
        stem = str(source_input["stem"])
        input_alt = escape(f"{kind} input {stem}")
        cells = [
            '<div class="cell">'
            f'<a href="{escape(source_input["file"])}">'
            f'<img src="{escape(thumbnail_filename(source_input["file"]))}" alt="{input_alt}"/></a>'
            f'<div class="caption"><b>{escape(stem)}</b> · source label {int(source_input["label"])}</div>'
            "</div>"
        ]
        for pass_num in range(1, passes + 1):
            record = records_by_key[(kind, stem, pass_num)]
            output_alt = escape(f"{kind} input {stem}, pass {pass_num} output")
            cells.append(
                '<div class="cell">'
                f'<a href="{escape(record["file"])}">'
                f'<img src="{escape(thumbnail_filename(record["file"]))}" alt="{output_alt}"/></a>'
                f'<div class="caption">{label_badge(label=int(record["returned_label"]))} '
                f"{escape(PATH_SHORT_TEXT[record['path']])}</div>"
                f'<div class="caption">{recipe_html(record=record)}</div>'
                "</div>"
            )
        rows.append(f'<div class="row" {grid_style}>{"".join(cells)}</div>')
    return f"<h2>{escape(section_title)}</h2>\n" + "\n".join(rows)


def reproduction_command(args: argparse.Namespace) -> str:
    """Command line that reproduces this run byte-identically (non-default flags only)."""
    parts = [
        "python src/preview_train_augmentation.py",
        f"--num-scenes {args.num_scenes}",
        f"--passes {args.passes}",
        f"--seed {args.seed}",
    ]
    if args.output_dir != AUGMENTATION_PREVIEW_OUTPUT_DIR:
        parts.append(f"--output-dir {args.output_dir}")
    if args.jpeg_quality != JPEG_QUALITY:
        parts.append(f"--jpeg-quality {args.jpeg_quality}")
    if args.moire_augment_prob != MOIRE_AUGMENT_PROB:
        parts.append(f"--moire-augment-prob {args.moire_augment_prob}")
    if args.moire_augment_types is not None:
        parts.append("--moire-augment-types " + " ".join(sorted(args.moire_augment_types)))
    if not args.group_augment:
        parts.append("--no-group-augment")
    if args.augment_groups is not None:
        parts.append("--augment-groups " + " ".join(sorted(args.augment_groups)))
    if args.augment_group_count is not None:
        parts.append("--augment-group-count " + " ".join(str(value) for value in args.augment_group_count))
    return " ".join(parts)


def render_html_document(
    generated_at: str,
    args: argparse.Namespace,
    type_names: list[str],
    probabilities: dict[str, float],
    group_names: list[str],
    group_count_range: tuple[int, int],
    inputs: list[dict[str, Any]],
    records: list[dict[str, Any]],
    stats: dict[str, Any],
) -> str:
    """Assemble the full self-contained HTML document for the saved preview."""
    records_by_key = {
        (record["source_kind"], record["source_stem"], record["pass"]): record for record in records
    }
    summary_rows = [
        ("generated", generated_at),
        ("script", "src/preview_train_augmentation.py"),
        (
            "dataset",
            f"first clean/moire source folder pair · {CLEAN_IMAGE_FOLDERS[0]} + {MOIRE_IMAGE_FOLDERS[0]} (paired scenes)",
        ),
        ("inputs", f"{args.num_scenes} clean + {args.num_scenes} real moire"),
        ("passes", f"{args.passes} over one dataset instance (rng advances like epochs)"),
        ("seed", str(args.seed)),
        ("augmentation class", "spatial_moire.baseline.augmentation.augmented_moire_dataset (transform=None)"),
        ("moire augment prob", str(args.moire_augment_prob)),
        ("moire types", ", ".join(f"{name} {probabilities[name]:.0%}" for name in type_names)),
        ("group augmentation", "on" if args.group_augment else "off (--no-group-augment)"),
        ("groups", f"{', '.join(group_names)} · {group_count_range[0]}-{group_count_range[1]} per draw"),
        ("jpeg quality", f"{args.jpeg_quality} full size / {AUGMENTATION_PREVIEW_THUMB_JPEG_QUALITY} thumbnails"),
    ]
    summary_table = "<table>\n" + "".join(
        f"<tr><th>{escape(name)}</th><td>{escape(value)}</td></tr>\n" for name, value in summary_rows
    ) + "</table>"
    notes = [
        "Every output below is exactly what augmented_moire_dataset.__getitem__ returns during training with "
        "transform=None, i.e. the augmented image before the torchvision train transform (RandomResizedCrop 224, "
        "flips, ColorJitter, ImageNet normalization). The moire synthesis and the group augmentation are identical "
        "to training.",
        "Labels are the labels training consumes: clean inputs that are moire-synthesized are relabeled 1, clean "
        "inputs that stay clean keep label 0, and real moire inputs keep label 1.",
        "One dataset instance serves all passes and its rng advances across them (like epochs), so every pass draws "
        "a fresh recipe for the same input.",
        "Recipes are captured with delegate-only wrappers around the training dispatch points (no extra rng draws, "
        "no pixel changes); a recorder-free same-seed replay of pass 1 reproduced every output pixel-for-pixel.",
        "Group augmentation is on by default in this preview; training opts in via --group-augment.",
        "Inputs are copied into this folder so the document is portable; click any image for the full-size JPEG. "
        "Full recipes (synthesis params, group augmenters and seeds) are in the JSON manifest.",
    ]
    notes_html = "".join(f"<li>{escape(note)}</li>\n" for note in notes)

    if args.moire_augment_prob > 0.0:
        conversion_line = (
            f"clean → moire conversions: {stats['num_converted_clean']} / {stats['num_clean_records']} clean fetches "
            f"({stats['num_converted_clean'] / max(stats['num_clean_records'], 1):.0%}; "
            f"expected {args.moire_augment_prob:.0%})"
        )
    else:
        conversion_line = "moiré synthesis disabled (augment_prob = 0): clean inputs can only receive groups"
    groups_line = (
        f"grouped fetches: {stats['num_grouped_draws']} · mean {stats['mean_groups_per_draw']:.2f} groups per draw "
        f"(draw range {group_count_range[0]}-{group_count_range[1]})"
    )
    total_records = max(int(stats["num_records"]), 1)
    path_rows = "".join(
        f"<tr><td>{escape(PATH_SHORT_TEXT.get(path_name, path_name))}</td>"
        f"<td>{count}</td><td>{count / total_records:.0%}</td></tr>\n"
        for path_name, count in sorted(stats["path_counts"].items())
    )
    type_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{int(stats['type_counts'].get(name, 0))}</td>"
        f"<td>{probabilities[name]:.0%}</td></tr>\n"
        for name in type_names
    )
    grouped_draws = max(int(stats["num_grouped_draws"]), 1)
    group_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{int(stats['group_counts'].get(name, 0))}</td>"
        f"<td>{int(stats['group_counts'].get(name, 0)) / grouped_draws:.0%}</td></tr>\n"
        for name in group_names
    )


    clean_section = render_scene_section(
        kind="clean",
        section_title=f"Clean inputs (label {CLEAN_LABEL})",
        inputs=inputs,
        records_by_key=records_by_key,
        passes=args.passes,
    )
    moire_section = render_scene_section(
        kind="moire",
        section_title=f"Real moiré inputs (label {MOIRE_LABEL})",
        inputs=inputs,
        records_by_key=records_by_key,
        passes=args.passes,
    )
    sections_html = clean_section + "\n" + moire_section
    reproduction = reproduction_command(args=args)
    footer_html = (
        "<footer>\n"
        f'<p><a href="{escape(AUGMENTATION_PREVIEW_MANIFEST_FILENAME)}">{escape(AUGMENTATION_PREVIEW_MANIFEST_FILENAME)}</a> '
        "holds every input, output and the full drawn recipes (synthesis params, group augmenters and seeds).</p>\n"
        f"<p>Reproduce byte-identically: <code>{escape(reproduction)}</code></p>\n"
        "</footer>\n"
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{escape(PAGE_TITLE)}</title>\n"
        f"<style>{PREVIEW_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>{escape(PAGE_TITLE)}</h1>\n"
        "<section>\n<h2>Configuration</h2>\n"
        f"{summary_table}"
        f'<ul class="notes">\n{notes_html}</ul>\n'
        "</section>\n"
        "<section>\n<h2>Draw statistics</h2>\n"
        f"<p>{escape(conversion_line)}<br>{escape(groups_line)}</p>\n"
        '<div class="tables">\n'
        "<table>\n<tr><th>path</th><th>count</th><th>share</th></tr>\n"
        f"{path_rows}</table>\n"
        "<table>\n<tr><th>moire type</th><th>drawn</th><th>expected</th></tr>\n"
        f"{type_rows}</table>\n"
        "<table>\n<tr><th>group</th><th>drawn</th><th>share of grouped fetches</th></tr>\n"
        f"{group_rows}</table>\n"
        "</div>\n"
        '<p class="note">Small sample: shares drift from the expected values; the manifest holds every draw.</p>\n'
        "</section>\n"
        f"{sections_html}\n"
        f"{footer_html}"
        "</body>\n"
        "</html>\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preview the training augmentation pipeline on clean and moire inputs "
            "and save the outputs for analysis."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=AUGMENTATION_PREVIEW_OUTPUT_DIR)
    parser.add_argument("--num-scenes", type=int, default=AUGMENTATION_PREVIEW_NUM_SCENES)
    parser.add_argument("--passes", type=int, default=AUGMENTATION_PREVIEW_PASSES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    parser.add_argument("--moire-augment-prob", type=float, default=MOIRE_AUGMENT_PROB)
    parser.add_argument("--moire-augment-types", nargs="+", choices=sorted(MOIRE_TYPES), default=None)
    parser.add_argument(
        "--no-group-augment",
        dest="group_augment",
        action="store_false",
        help="disable the capture-style group augmentation (on by default; training opts in via --group-augment)",
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
        help="inclusive range of augmentation groups applied per augmented image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.passes < 1:
        raise SystemExit("--passes must be at least 1")
    selection_rng = np.random.default_rng(seed=args.seed)
    pairs = select_scene_pairs(num_scenes=args.num_scenes, rng=selection_rng)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = samples_from_pairs(pairs=pairs)
    # samples_from_pairs emits the moire sample before the clean sample of each pair.
    scene_kinds = [kind for _pair in pairs for kind in ("moire", "clean")]
    scene_stems = [Path(sample["path"]).stem for sample in samples]
    print(
        f"previewing {args.num_scenes} paired scenes ({args.num_scenes} clean + {args.num_scenes} real moire) "
        f"x {args.passes} passes with seed {args.seed}",
        flush=True,
    )

    inputs = copy_inputs(samples=samples, scene_kinds=scene_kinds, scene_stems=scene_stems, output_dir=args.output_dir)

    type_names = list(MOIRE_TYPES) if args.moire_augment_types is None else sorted(args.moire_augment_types)
    group_names = list(AUGMENTATION_GROUPS) if args.augment_groups is None else sorted(args.augment_groups)
    group_count_range = (
        GROUP_AUGMENT_COUNT_RANGE if args.augment_group_count is None else tuple(args.augment_group_count)
    )
    dataset_kwargs: dict[str, Any] = {
        "samples": samples,
        "transform": None,
        "augment_prob": args.moire_augment_prob,
        "type_names": type_names,
        "seed": args.seed,
        "group_augment": args.group_augment,
        "group_names": group_names,
        "num_groups_range": group_count_range,
    }
    dataset = augmented_moire_dataset(**dataset_kwargs)
    records, pass_one = generate_outputs(
        dataset=dataset,
        samples=samples,
        scene_kinds=scene_kinds,
        scene_stems=scene_stems,
        output_dir=args.output_dir,
        passes=args.passes,
        jpeg_quality=args.jpeg_quality,
    )
    verify_recording_is_pixel_neutral(dataset_kwargs=dataset_kwargs, pass_one=pass_one)

    stats = compute_statistics(records=records)
    probabilities = type_probabilities(type_names=type_names)
    manifest = {
        "seed": args.seed,
        "clean_image_folders": [str(folder) for folder in CLEAN_IMAGE_FOLDERS],
        "moire_image_folders": [str(folder) for folder in MOIRE_IMAGE_FOLDERS],
        "selected_scenes_folder": str(CLEAN_IMAGE_FOLDERS[0]),
        "selected_scenes": [gt_path.name for _moire_path, gt_path in pairs],
        "config": {
            "num_scenes": args.num_scenes,
            "passes": args.passes,
            "augment_prob": args.moire_augment_prob,
            "type_names": type_names,
            "type_probabilities": probabilities,
            "group_augment": args.group_augment,
            "group_names": group_names,
            "num_groups_range": list(group_count_range),
            "jpeg_quality": args.jpeg_quality,
        },
        "verification": {"replayed_pass": 1, "pixel_identical": True},
        "inputs": inputs,
        "statistics": stats,
        "entries": records,
    }
    manifest_path = args.output_dir / AUGMENTATION_PREVIEW_MANIFEST_FILENAME
    manifest_path.write_text(data=json.dumps(manifest, indent=JSON_INDENT), encoding=JSON_ENCODING)

    html = render_html_document(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        args=args,
        type_names=type_names,
        probabilities=probabilities,
        group_names=group_names,
        group_count_range=group_count_range,
        inputs=inputs,
        records=records,
        stats=stats,
    )
    (args.output_dir / AUGMENTATION_PREVIEW_HTML_FILENAME).write_text(data=html, encoding=JSON_ENCODING)
    print(
        f"wrote {len(inputs)} inputs, {len(records)} augmented outputs, {AUGMENTATION_PREVIEW_MANIFEST_FILENAME} "
        f"and {AUGMENTATION_PREVIEW_HTML_FILENAME} to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()