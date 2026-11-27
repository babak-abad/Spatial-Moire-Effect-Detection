"""Scratch validation of imgaug 0.4.0 under numpy 2.5 / Python 3.12 (deleted after use)."""

from __future__ import annotations

import inspect
import traceback

from collections.abc import Callable

import numpy as np

# numpy 2.x removed np.sctypes/np.bool/np.complex; imgaug 0.4.0 reads them
# (imgaug.py:45-47 at import time; meta.py:3368, polys.py:951, parameters.py:2198/3154 at runtime).
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "bool": [np.bool_],
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
    }
np.bool = np.bool_  # noqa: A001
np.complex = np.complex128  # noqa: A001

import imgaug as ia  # noqa: E402
from imgaug import augmenters as iaa  # noqa: E402

FAILURES: list[str] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" | {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def check_augmenter(name: str, factory: Callable[[], object], image: np.ndarray) -> None:
    try:
        augmenter = factory()
        out = augmenter.augment_image(image)
        ok = isinstance(out, np.ndarray) and out.shape == image.shape and out.dtype == image.dtype
        report(name, ok, f"shape={out.shape} dtype={out.dtype} range=[{float(out.min()):.3f}, {float(out.max()):.3f}]")
    except Exception as error:  # noqa: BLE001
        report(name, False, f"{type(error).__name__}: {error}")
        traceback.print_exc()


def main() -> None:
    print(f"imgaug={ia.__version__} numpy={np.__version__}")
    rng = np.random.default_rng(seed=1)
    image_float = np.clip(rng.normal(loc=0.5, scale=0.2, size=(64, 80, 3)), 0.0, 1.0).astype(np.float32)
    image_uint8 = (image_float * 255.0).round().astype(np.uint8)

    report("import imgaug with numpy2 aliases shim", hasattr(ia, "NP_FLOAT_TYPES"))
    for class_name in (
        "Affine",
        "GaussianBlur",
        "AdditiveGaussianNoise",
        "SaltAndPepper",
        "Salt",
        "Dropout",
        "CoarseDropout",
        "CoarseSalt",
        "CoarseSaltAndPepper",
        "Multiply",
        "MultiplyElementwise",
        "WithChannels",
        "ReplaceElementwise",
    ):
        cls = getattr(iaa, class_name, None)
        report(f"iaa.{class_name} exists", cls is not None, str(inspect.signature(cls.__init__)) if cls else "")

    check_augmenter("Affine rotate", lambda: iaa.Affine(rotate=17.0, order=1, mode="constant", cval=0.37, backend="skimage"), image_float)
    check_augmenter("Affine zoom-in", lambda: iaa.Affine(scale=1.25, order=1, mode="constant", cval=0.37, backend="skimage"), image_float)
    check_augmenter("Affine zoom-out", lambda: iaa.Affine(scale=0.85, order=1, mode="constant", cval=0.37, backend="skimage"), image_float)
    check_augmenter("GaussianBlur", lambda: iaa.GaussianBlur(sigma=1.2), image_float)
    check_augmenter("AdditiveGaussianNoise", lambda: iaa.AdditiveGaussianNoise(scale=0.05, seed=11), image_float)
    check_augmenter("SaltAndPepper", lambda: iaa.SaltAndPepper(p=0.1, seed=11), image_float)
    check_augmenter("Dropout", lambda: iaa.Dropout(p=0.08, seed=11), image_float)
    check_augmenter("CoarseDropout", lambda: iaa.CoarseDropout(p=0.3, size_percent=0.15, seed=11), image_float)
    check_augmenter("Dropout per_channel", lambda: iaa.Dropout(p=0.4, per_channel=True, seed=11), image_float)
    check_augmenter(
        "CoarseDropout per_channel", lambda: iaa.CoarseDropout(p=0.4, size_percent=0.2, per_channel=True, seed=11), image_float
    )
    check_augmenter("MultiplyElementwise", lambda: iaa.MultiplyElementwise(mul=(0.85, 1.15), seed=11), image_float)
    check_augmenter(
        "WithChannels+Dropout",
        lambda: iaa.WithChannels(channels=[1], children=iaa.Dropout(p=1.0, seed=11)),
        image_float,
    )
    check_augmenter(
        "WithChannels+ReplaceElementwise",
        lambda: iaa.WithChannels(channels=[1], children=iaa.ReplaceElementwise(mask=1.0, replacement=(0.0, 1.0), seed=11)),
        image_float,
    )
    check_augmenter("uint8 SaltAndPepper", lambda: iaa.SaltAndPepper(p=0.1, seed=5), image_uint8)

    constant = np.full(shape=(64, 80, 3), fill_value=0.5, dtype=np.float32)
    white = iaa.ReplaceElementwise(mask=0.5, replacement=1.0, seed=77).augment_image(constant)
    report(
        "ReplaceElementwise replacement=1.0 is exact on float",
        float(white.max()) == 1.0 and float(white.min()) == 0.5,
        f"range=[{float(white.min()):.3f}, {float(white.max()):.3f}]",
    )
    channel_out = iaa.WithChannels(channels=[1], children=iaa.Dropout(p=1.0, seed=11)).augment_image(image_float)
    report(
        "WithChannels zeroes only the target channel",
        np.array_equal(channel_out[:, :, 1], np.zeros_like(channel_out[:, :, 1]))
        and np.array_equal(channel_out[:, :, 0], image_float[:, :, 0])
        and np.array_equal(channel_out[:, :, 2], image_float[:, :, 2]),
    )
    perchan = iaa.Dropout(p=0.4, per_channel=True, seed=5).augment_image(image_float)
    zeros_per_channel = [int((perchan[:, :, c] == 0.0).sum()) for c in range(3)]
    report("Dropout per_channel uses independent masks", len(set(zeros_per_channel)) > 1, f"zeros={zeros_per_channel}")

    out_a = iaa.Dropout(p=0.2, seed=999).augment_image(image_float)
    out_b = iaa.Dropout(p=0.2, seed=999).augment_image(image_float)
    out_c = iaa.Dropout(p=0.2, seed=1234).augment_image(image_float)
    report("fresh instance + same seed is deterministic", np.array_equal(out_a, out_b))
    report("different seeds differ", not np.array_equal(out_a, out_c))

    noisy = iaa.AdditiveGaussianNoise(scale=0.2, seed=31).augment_image(constant)
    report(
        "AdditiveGaussianNoise scale is in [0,1] units on float",
        abs(float(noisy.std()) - 0.2) < 0.05,
        f"std={float(noisy.std()):.4f}",
    )

    rotated = iaa.Affine(rotate=25.0, order=1, mode="constant", cval=0.37, backend="skimage").augment_image(image_float)
    report("Affine scalar cval fill on float (skimage backend)", np.allclose(rotated[0, 0], 0.37, atol=0.02), f"corner={rotated[0, 0]}")

    speckled = iaa.MultiplyElementwise(mul=(0.9, 1.1), seed=13).augment_image(constant)
    report(
        "MultiplyElementwise speckle in mul range",
        float(speckled.min()) >= 0.44 and float(speckled.max()) <= 0.56,
        f"range=[{float(speckled.min()):.3f}, {float(speckled.max()):.3f}]",
    )

    print(f"\n{len(FAILURES)} failures: {FAILURES}" if FAILURES else "\nall checks passed")
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
