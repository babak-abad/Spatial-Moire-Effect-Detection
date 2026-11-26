"""On-the-fly moire augmentation for clean images in the training dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import get_worker_info
from torchvision import transforms

from spatial_moire.baseline.data import CLEAN_LABEL, MOIRE_LABEL, moire_dataset
from spatial_moire.synthesis.augment import MOIRE_GENERATORS, generator_callable


class augmented_moire_dataset(moire_dataset):
    """moire_dataset that turns a random share of clean images into moire.

    Each clean image is moire-augmented in memory with probability
    augment_prob and then carries the moire label, so an augmented image can
    never be returned as clean. Real moire images are never augmented and
    source files are never modified.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        transform: transforms.Compose | None = None,
        augment_prob: float = 0.5,
        generator_names: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(samples=samples, transform=transform)
        if not 0.0 <= augment_prob <= 1.0:
            raise ValueError(f"augment_prob must be in [0.0, 1.0], got {augment_prob}")
        selected_names = sorted(MOIRE_GENERATORS) if generator_names is None else sorted(generator_names)
        unknown_names = [name for name in selected_names if name not in MOIRE_GENERATORS]
        if unknown_names:
            raise ValueError(f"unknown moire generator names: {unknown_names}")
        self.augment_prob = augment_prob
        self.seed = seed
        self._generator_items: list[tuple[str, generator_callable]] = [
            (name, MOIRE_GENERATORS[name]) for name in selected_names
        ]
        self._worker_rngs: dict[int, np.random.Generator] = {}

    def _rng(self) -> np.random.Generator:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = self._worker_rngs.get(worker_id)
        if rng is None:
            rng = np.random.default_rng(seed=self.seed + worker_id)
            self._worker_rngs[worker_id] = rng
        return rng

    def _augment(self, image: Image.Image, rng: np.random.Generator) -> Image.Image:
        generator = self._generator_items[int(rng.integers(low=0, high=len(self._generator_items)))][1]
        array = np.asarray(image, dtype=np.float32) / 255.0
        augmented, _params = generator(image=array, rng=rng)
        data = (np.clip(a=augmented, a_min=0.0, a_max=1.0) * 255.0).round().astype(dtype=np.uint8)
        return Image.fromarray(obj=data)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        sample = self.samples[index]
        image = Image.open(sample["path"]).convert("RGB")
        label = int(sample["label"])
        if label == CLEAN_LABEL and self._generator_items and self.augment_prob > 0.0:
            rng = self._rng()
            if rng.random() < self.augment_prob:
                image = self._augment(image=image, rng=rng)
                label = MOIRE_LABEL
        if self.transform is not None:
            image = self.transform(image)
        return image, label
