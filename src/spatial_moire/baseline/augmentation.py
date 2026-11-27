"""On-the-fly moire augmentation for clean images in the training dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import get_worker_info
from torchvision import transforms

from spatial_moire.baseline.data import CLEAN_LABEL, MOIRE_LABEL, moire_dataset
from config import DEFAULT_SEED, GROUP_AUGMENT_COUNT_RANGE, IMAGE_COLOR_MODE, MOIRE_AUGMENT_PROB, PIXEL_VALUE_MAX
from spatial_moire.moire_types import MOIRE_TYPES, generator_names_for_type, sample_moire_type
from spatial_moire.synthesis.augment import MOIRE_GENERATORS
from spatial_moire.synthesis.groups import apply_group_augmentation, normalized_group_count_range, selected_group_names


class augmented_moire_dataset(moire_dataset):
    """moire_dataset that turns a random share of clean images into moire.

    Each clean image is moire-augmented in memory with probability
    augment_prob and then carries the moire label, so an augmented image can
    never be returned as clean. The moire type of every augmentation is drawn
    with possibility weight / total_weight from the selected types (all nine
    registry types by default) and rendered by one of that type's mapped
    synthesis generators. With group_augment=True every image is
    additionally corrupted with a random subset of the capture-style
    augmentation groups from spatial_moire.synthesis.groups while the label
    keeps following the moire decision alone: moire-augmented clean images
    after synthesis, and real moire images plus clean images that stay clean
    straight from the label decision, since they skip moire synthesis but
    enter the same group_augment branch. Source files are never modified.

    With balanced=True the dataset instead serves the interleaved virtual index
    space of balanced_moire_dataset: even indices return a clean image with the
    clean label (the moire draw never fires on them) and odd indices return a
    moire image with the moire label — a real moire source passed through or,
    with probability augment_prob, a clean source with forced moire synthesis.
    augment_prob thereby becomes the share of the moire half that is
    synthesized, and any batch of even size drawn from sequential or
    balanced_shuffle_sampler order holds exactly half clean and half moire
    items regardless of the source label distribution.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        transform: transforms.Compose | None = None,
        augment_prob: float = MOIRE_AUGMENT_PROB,
        type_names: list[str] | None = None,
        seed: int = DEFAULT_SEED,
        group_augment: bool = False,
        group_names: list[str] | None = None,
        num_groups_range: tuple[int, int] = GROUP_AUGMENT_COUNT_RANGE,
        balanced: bool = False,
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
        self.group_augment = group_augment
        self.group_names = selected_group_names(group_subset=group_names)
        self.num_groups_range = normalized_group_count_range(num_groups_range=num_groups_range)
        self._worker_rngs: dict[int, np.random.Generator] = {}
        self.balanced = balanced
        if balanced:
            self.clean_pool = [sample for sample in samples if int(sample["label"]) == CLEAN_LABEL]
            self.moire_pool = [sample for sample in samples if int(sample["label"]) == MOIRE_LABEL]
            synthesis_available = bool(self.type_names) and self.augment_prob > 0.0
            if not self.clean_pool:
                raise ValueError(f"balanced=True requires at least one clean sample, got {len(self.clean_pool)}")
            if not self.moire_pool and not synthesis_available:
                raise ValueError(
                    "balanced=True requires moire samples or moire synthesis (augment_prob > 0.0 with at least one moire type)"
                )
            self.num_slots = max(len(self.clean_pool), len(self.moire_pool))

    def _rng(self) -> np.random.Generator:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = self._worker_rngs.get(worker_id)
        if rng is None:
            rng = np.random.default_rng(seed=self.seed + worker_id)
            self._worker_rngs[worker_id] = rng
        return rng

    def _apply_groups(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        augmented, _group_params = apply_group_augmentation(
            image=image,
            rng=rng,
            group_subset=self.group_names,
            num_groups_range=self.num_groups_range,
        )
        return augmented

    def _to_pil(self, array: np.ndarray) -> Image.Image:
        data = (np.clip(a=array, a_min=0.0, a_max=1.0) * PIXEL_VALUE_MAX).round().astype(dtype=np.uint8)
        return Image.fromarray(obj=data)

    def _group_only(self, image: Image.Image, rng: np.random.Generator) -> Image.Image:
        array = np.asarray(image, dtype=np.float32) / PIXEL_VALUE_MAX
        grouped = self._apply_groups(image=array, rng=rng)
        return self._to_pil(array=grouped)

    def _augment(self, image: Image.Image, rng: np.random.Generator) -> Image.Image:
        type_name = sample_moire_type(rng=rng, type_names=self.type_names)
        generator_name = str(rng.choice(a=generator_names_for_type(type_name=type_name)))
        array = np.asarray(image, dtype=np.float32) / PIXEL_VALUE_MAX
        augmented, _params = MOIRE_GENERATORS[generator_name](image=array, rng=rng)
        if self.group_augment:
            augmented = self._apply_groups(image=augmented, rng=rng)
        return self._to_pil(array=augmented)

    def __len__(self) -> int:
        if self.balanced:
            return 2 * self.num_slots
        return len(self.samples)

    def _getitem_balanced(self, index: int) -> tuple[Any, int]:
        """Serve the interleaved index space: parity fixes the returned label.

        Even indices return a clean image (label 0) — moire synthesis never
        fires on them, group augmentation follows the flag alone. Odd indices
        return a moire image (label 1): with probability augment_prob a clean
        source is moire-augmented, otherwise a real moire source passes through
        unchanged apart from the group augmentation.
        """
        slot, want_moire = divmod(index, 2)
        rng = self._rng()
        if want_moire:
            synthesis_available = bool(self.type_names) and self.augment_prob > 0.0
            if not self.moire_pool or (synthesis_available and rng.random() < self.augment_prob):
                source = self.clean_pool[slot % len(self.clean_pool)]
                image = Image.open(source["path"]).convert(mode=IMAGE_COLOR_MODE)
                image = self._augment(image=image, rng=rng)
            else:
                source = self.moire_pool[slot % len(self.moire_pool)]
                image = Image.open(source["path"]).convert(mode=IMAGE_COLOR_MODE)
                if self.group_augment:
                    image = self._group_only(image=image, rng=rng)
            label = MOIRE_LABEL
        else:
            source = self.clean_pool[slot % len(self.clean_pool)]
            image = Image.open(source["path"]).convert(mode=IMAGE_COLOR_MODE)
            if self.group_augment:
                image = self._group_only(image=image, rng=rng)
            label = CLEAN_LABEL
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __getitem__(self, index: int) -> tuple[Any, int]:
        if self.balanced:
            return self._getitem_balanced(index=index)
        sample = self.samples[index]
        image = Image.open(sample["path"]).convert(mode=IMAGE_COLOR_MODE)
        label = int(sample["label"])
        if label == CLEAN_LABEL and self.type_names and self.augment_prob > 0.0:
            rng = self._rng()
            if rng.random() < self.augment_prob:
                image = self._augment(image=image, rng=rng)
                label = MOIRE_LABEL
            elif self.group_augment:
                image = self._group_only(image=image, rng=rng)
        elif self.group_augment:
            image = self._group_only(image=image, rng=self._rng())
        if self.transform is not None:
            image = self.transform(image)
        return image, label
