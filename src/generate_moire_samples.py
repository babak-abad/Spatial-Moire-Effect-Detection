"""Generate moire-augmented sample images from clean inputs into the outputs folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spatial_moire.synthesis.augment import MOIRE_GENERATORS  # noqa: E402

DEFAULT_SOURCE_DIR = REPO_ROOT / "datasets" / "LCDMoire" / "my_dataset_final" / "test" / "gt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def load_image(path: Path) -> np.ndarray:
    image = Image.open(fp=path).convert(mode="RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def save_image(array: np.ndarray, path: Path, jpeg_quality: int) -> None:
    data = (np.clip(a=array, a_min=0.0, a_max=1.0) * 255.0).round().astype(dtype=np.uint8)
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
    figure, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(columns * 2.4, rows * 2.4), squeeze=False)
    for row in range(rows):
        axes[row][0].imshow(X=clean_images[row])
        axes[row][0].set_ylabel(ylabel=f"source {row + 1}", fontsize=9)
        for column, type_name in enumerate(type_names, start=1):
            axes[row][column].imshow(X=moire_by_type[type_name][row])
        for axis in axes[row]:
            axis.set_xticks(ticks=[])
            axis.set_yticks(ticks=[])
    labels = ["clean", *type_names]
    for column, label in enumerate(labels):
        axes[0][column].set_title(label=label, fontsize=10)
    figure.tight_layout()
    figure.savefig(fname=output_path, dpi=110)
    plt.close(fig=figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate moire-augmented samples from clean images.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument(
        "--types",
        nargs="+",
        choices=list(MOIRE_GENERATORS.keys()),
        default=list(MOIRE_GENERATORS.keys()),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=95)
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

    manifest: list[dict] = []
    clean_images: list[np.ndarray] = []
    moire_by_type: dict[str, list[np.ndarray]] = {type_name: [] for type_name in args.types}
    for path in image_paths:
        image = load_image(path=path)
        clean_images.append(image)
        for type_name in args.types:
            generator = MOIRE_GENERATORS[type_name]
            for variant in range(1, args.variants + 1):
                moire_image, params = generator(image=image, rng=rng)
                output_name = f"moire_{type_name}__{path.stem}_v{variant:02d}.jpg"
                save_image(array=moire_image, path=args.output_dir / output_name, jpeg_quality=args.jpeg_quality)
                manifest.append(
                    {
                        "file": output_name,
                        "source": str(path),
                        "type": type_name,
                        "variant": variant,
                        "params": params,
                    }
                )
                print(f"wrote {output_name}", flush=True)
            moire_by_type[type_name].append(moire_image)
    manifest_path = args.output_dir / "moire_synthesis_params.json"
    manifest_path.write_text(
        data=json.dumps({"seed": args.seed, "num_images": len(image_paths), "entries": manifest}, indent=2),
        encoding="utf-8",
    )
    build_preview_grid(
        clean_images=clean_images,
        moire_by_type=moire_by_type,
        output_path=args.output_dir / "preview_grid.png",
    )
    print(f"wrote {len(manifest)} samples, preview grid and params manifest to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()