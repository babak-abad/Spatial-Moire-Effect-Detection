"""Generate moire-augmented sample images from clean inputs into the outputs folder."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from spatial_moire.config import (  # noqa: E402
    DEFAULT_SAMPLING_MODE,
    DEFAULT_SEED,
    DEFAULT_SOURCE_DIR,
    DRAWN_PREVIEW_ROW_HEIGHT,
    IMAGE_EXTENSIONS,
    JPEG_QUALITY,
    JSON_ENCODING,
    JSON_INDENT,
    MATPLOTLIB_BACKEND,
    MAX_DRAWN_PREVIEW_IMAGES,
    NUM_SAMPLE_IMAGES,
    NUM_VARIANTS,
    PIXEL_VALUE_MAX,
    PREVIEW_CELL_SIZE,
    PREVIEW_DPI,
    PREVIEW_DRAWN_FILENAME,
    PREVIEW_GRID_FILENAME,
    PREVIEW_LABEL_FONTSIZE,
    PREVIEW_TITLE_FONTSIZE,
    SAMPLES_OUTPUT_DIR,
    SAMPLING_MODES,
    SYNTHESIS_MANIFEST_FILENAME,
)

matplotlib.use(MATPLOTLIB_BACKEND)
import matplotlib.pyplot as plt  # noqa: E402

from spatial_moire.moire_types import (  # noqa: E402
    MOIRE_TYPES,
    generator_names_for_type,
    sample_moire_type,
    type_probabilities,
)
from spatial_moire.synthesis.augment import MOIRE_GENERATORS  # noqa: E402


def load_image(path: Path) -> np.ndarray:
    image = Image.open(fp=path).convert(mode="RGB")
    return np.asarray(image, dtype=np.float32) / PIXEL_VALUE_MAX


def save_image(array: np.ndarray, path: Path, jpeg_quality: int) -> None:
    data = (np.clip(a=array, a_min=0.0, a_max=1.0) * PIXEL_VALUE_MAX).round().astype(dtype=np.uint8)
    Image.fromarray(obj=data).save(fp=path, quality=jpeg_quality)


def build_preview_grid(
    clean_images: list[np.ndarray],
    moire_by_type: dict[str, list[np.ndarray]],
    output_path: Path,
) -> None:
    """Save a rows=sources, columns=clean+types preview grid."""
    type_names = list(moire_by_type.keys())
    rows = len(clean_images)
    columns = 1 + len(type_names)
    figure, axes = plt.subplots(
        nrows=rows,
        ncols=columns,
        figsize=(columns * PREVIEW_CELL_SIZE, rows * PREVIEW_CELL_SIZE),
        squeeze=False,
    )
    for row in range(rows):
        axes[row][0].imshow(X=clean_images[row])
        axes[row][0].set_ylabel(ylabel=f"source {row + 1}", fontsize=PREVIEW_LABEL_FONTSIZE)
        for column, type_name in enumerate(type_names, start=1):
            axes[row][column].imshow(X=moire_by_type[type_name][row])
        for axis in axes[row]:
            axis.set_xticks(ticks=[])
            axis.set_yticks(ticks=[])
    labels = ["clean", *type_names]
    for column, label in enumerate(labels):
        axes[0][column].set_title(label=label, fontsize=PREVIEW_TITLE_FONTSIZE)
    figure.tight_layout()
    figure.savefig(fname=output_path, dpi=PREVIEW_DPI)
    plt.close(fig=figure)


def render_moire_type(
    image: np.ndarray,
    type_name: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str, dict]:
    """Run one of the type's mapped generators and return (image, generator name, params)."""
    generator_name = str(rng.choice(a=generator_names_for_type(type_name=type_name)))
    moire_image, params = MOIRE_GENERATORS[generator_name](image=image, rng=rng)
    return moire_image, generator_name, params


def build_drawn_preview(images: list[np.ndarray], labels: list[str], output_path: Path) -> None:
    """Save a single-row preview of weighted-mode samples labeled by drawn type."""
    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(images),
        figsize=(len(images) * PREVIEW_CELL_SIZE, DRAWN_PREVIEW_ROW_HEIGHT),
        squeeze=False,
    )
    for column, (image, label) in enumerate(zip(images, labels)):
        axes[0][column].imshow(X=image)
        axes[0][column].set_title(label=label, fontsize=PREVIEW_LABEL_FONTSIZE)
        axes[0][column].set_xticks(ticks=[])
        axes[0][column].set_yticks(ticks=[])
    figure.tight_layout()
    figure.savefig(fname=output_path, dpi=PREVIEW_DPI)
    plt.close(fig=figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate moire-augmented samples from clean images.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=SAMPLES_OUTPUT_DIR)
    parser.add_argument("--num-images", type=int, default=NUM_SAMPLE_IMAGES)
    parser.add_argument("--variants", type=int, default=NUM_VARIANTS)
    parser.add_argument(
        "--types",
        nargs="+",
        choices=sorted(MOIRE_TYPES),
        default=list(MOIRE_TYPES),
    )
    parser.add_argument(
        "--sampling",
        choices=SAMPLING_MODES,
        default=DEFAULT_SAMPLING_MODE,
        help="grid: one sample per selected type per image; weighted: draw each type by weight / total_weight",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(seed=args.seed)
    image_paths = sorted(path for path in args.source_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise SystemExit(f"no source images found in {args.source_dir}")
    if args.num_images < len(image_paths):
        chosen = rng.permutation(len(image_paths))[: args.num_images]
        image_paths = [image_paths[index] for index in sorted(chosen)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    probabilities = type_probabilities(type_names=args.types)
    manifest: list[dict] = []
    clean_images: list[np.ndarray] = []
    moire_by_type: dict[str, list[np.ndarray]] = {type_name: [] for type_name in args.types}
    drawn_images: list[np.ndarray] = []
    drawn_types: list[str] = []
    for path in image_paths:
        image = load_image(path=path)
        clean_images.append(image)
        if args.sampling == "grid":
            for type_name in args.types:
                for variant in range(1, args.variants + 1):
                    moire_image, generator_name, params = render_moire_type(
                        image=image,
                        type_name=type_name,
                        rng=rng,
                    )
                    output_name = f"moire_{type_name}__{path.stem}_v{variant:02d}.jpg"
                    save_image(array=moire_image, path=args.output_dir / output_name, jpeg_quality=args.jpeg_quality)
                    manifest.append(
                        {
                            "file": output_name,
                            "source": str(path),
                            "type": type_name,
                            "generator": generator_name,
                            "variant": variant,
                            "probability": probabilities[type_name],
                            "params": params,
                        }
                    )
                    print(f"wrote {output_name}", flush=True)
                moire_by_type[type_name].append(moire_image)
        else:
            for variant in range(1, args.variants + 1):
                type_name = sample_moire_type(rng=rng, type_names=args.types)
                moire_image, generator_name, params = render_moire_type(
                    image=image,
                    type_name=type_name,
                    rng=rng,
                )
                output_name = f"moire_{type_name}__{path.stem}_v{variant:02d}.jpg"
                save_image(array=moire_image, path=args.output_dir / output_name, jpeg_quality=args.jpeg_quality)
                manifest.append(
                    {
                        "file": output_name,
                        "source": str(path),
                        "type": type_name,
                        "generator": generator_name,
                        "variant": variant,
                        "probability": probabilities[type_name],
                        "params": params,
                    }
                )
                drawn_images.append(moire_image)
                drawn_types.append(type_name)
                print(f"wrote {output_name}", flush=True)
    manifest_path = args.output_dir / SYNTHESIS_MANIFEST_FILENAME
    manifest_path.write_text(
        data=json.dumps(
            {
                "seed": args.seed,
                "sampling": args.sampling,
                "num_images": len(image_paths),
                "type_probabilities": probabilities,
                "entries": manifest,
            },
            indent=JSON_INDENT,
        ),
        encoding=JSON_ENCODING,
    )
    if args.sampling == "grid":
        build_preview_grid(
            clean_images=clean_images,
            moire_by_type=moire_by_type,
            output_path=args.output_dir / PREVIEW_GRID_FILENAME,
        )
    else:
        preview_count = min(MAX_DRAWN_PREVIEW_IMAGES, len(drawn_images))
        if preview_count > 0:
            build_drawn_preview(
                images=drawn_images[:preview_count],
                labels=drawn_types[:preview_count],
                output_path=args.output_dir / PREVIEW_DRAWN_FILENAME,
            )
    type_counts = Counter(entry["type"] for entry in manifest)
    distribution = ", ".join(
        f"{name} {type_counts.get(name, 0)} ({probabilities[name]:.0%})" for name in probabilities
    )
    print(
        f"wrote {len(manifest)} samples via {args.sampling} sampling [{distribution}], "
        f"preview and params manifest to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()