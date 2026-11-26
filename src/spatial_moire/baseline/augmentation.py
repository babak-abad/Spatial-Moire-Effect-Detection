"""On-the-fly moire augmentation for clean images in the training dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import get_worker_info
from torchvision import transforms

from spatial_moire.baseline.data import CLEAN_LABEL, MOIRE_LABEL, moire_dataset
from config import DEFAULT_SEED, IMAGE_COLOR_MODE, MOIRE_AUGMENT_PROB, PIXEL_VALUE_MAX
from spatial_moire.moire_types import MOIRE_TYPES, generator_names_for_type, sample_moire_type
from spatial_moire.synthesis.augment import MOIRE_GENERATORS


class augmented_moire_dataset(moire_dataset):
    """moire_dataset that turns a random share of clean images into moire.

    Each clean image is moire-augmented in memory with probability
    augment_prob and then carries the moire label, so an augmented image can
    never be returned as clean. The moire type of every augmentation is drawn
    with possibility weight / total_weight from the selected types (all nine
    registry types by default) and rendered by one of that type's mapped
    synthesis generators. Real moire images are never augmented and source
    files are never modified.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        transform: transforms.Compose | None = None,
        augment_prob: float = MOIRE_AUGMENT_PROB,
        type_names: list[str] | None = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        super().__init__(samples=samples, transform=transform)
        if not 0.0 <= augment_prob <= 1.0:
            raise ValueError(f"augment_prob must be in [0.0, 1.0], got {augment_prob}")
        selected_names = list(MOIRE_TYPES) if type_names is None else list(type_names)
        unknown_names = [name for name in selected_names if name not in MOIRE_TYPES]
        if unknown_names:
            raise ValueError(f"unknown moire type names: {unknown_names}")
        referenced_generators = {
            generator_name
            for name in selected_names
            for generator_name in generator_names_for_type(type_name=name)
        }
        missing_generators = sorted(referenced_generators - set(MOIRE_GENERATORS))
        if missing_generators:
            raise ValueError(f"moire types reference unknown generators: {missing_generators}")
        self.augment_prob = augment_prob
        self.seed = seed
        self.type_names = selected_names
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
        type_name = sample_moire_type(rng=rng, type_names=self.type_names)
        generator_name = str(rng.choice(a=generator_names_for_type(type_name=type_name)))
        array = np.asarray(image, dtype=np.float32) / PIXEL_VALUE_MAX
        augmented, _params = MOIRE_GENERATORS[generator_name](image=array, rng=rng)
        data = (np.clip(a=augmented, a_min=0.0, a_max=1.0) * PIXEL_VALUE_MAX).round().astype(dtype=np.uint8)
        return Image.fromarray(obj=data)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        sample = self.samples[index]
        image = Image.open(sample["path"]).convert(mode=IMAGE_COLOR_MODE)
        label = int(sample["label"])
        if label == CLEAN_LABEL and self.type_names and self.augment_prob > 0.0:
            rng = self._rng()
            if rng.random() < self.augment_prob:
                image = self._augment(image=image, rng=rng)
                label = MOIRE_LABEL
        if self.transform is not None:
            image = self.transform(image)
        return image, label
