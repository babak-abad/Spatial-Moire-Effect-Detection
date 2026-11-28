"""Class-aware, label-preserving on-the-fly augmentation for the training dataset."""

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
    """Label-preserving, class-aware training augmentation shared by both classes.

    Clean fetches (label 0) only ever receive the label-neutral capture-style
    group augmentation (when group_augment=True) — moire synthesis is
    structurally unreachable on the clean path, so a clean image can never
    become moire inside the training pipeline. Moire fetches (label 1) may
    gain, with probability moire_augment_prob, one additional moire pattern
    drawn with possibility weight / total_weight from the selected types (all
    nine registry types by default) and rendered by one of that type's mapped
    synthesis generators, followed by the same group augmentation. The label
    of a fetch always equals the label of its source sample. Source files are
    never modified.

    With balanced=True the dataset serves the interleaved virtual index space
    of balanced_moire_dataset over the real class pools only: even indices
    return a clean fetch (label 0), odd indices a moire fetch (label 1), both
    cycling modulo their pool size, so any batch of even size drawn from
    sequential or balanced_shuffle_sampler order holds exactly half clean and
    half moire items. There is no synthesis fallback — an empty moire pool
    raises ValueError. With balanced=False the samples list is walked in
    order.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        transform: transforms.Compose | None = None,
        moire_augment_prob: float = MOIRE_AUGMENT_PROB,
        type_names: list[str] | None = None,
        seed: int = DEFAULT_SEED,
        group_augment: bool = False,
        group_names: list[str] | None = None,
        num_groups_range: tuple[int, int] = GROUP_AUGMENT_COUNT_RANGE,
        balanced: bool = False,
    ) -> None:
        super().__init__(samples=samples, transform=transform)
        if not 0.0 <= moire_augment_prob <= 1.0:
            raise ValueError(f"moire_augment_prob must be in [0.0, 1.0], got {moire_augment_prob}")
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
        self.moire_augment_prob = moire_augment_prob
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
            if not self.clean_pool:
                raise ValueError(f"balanced=True requires at least one clean sample, got {len(self.clean_pool)}")
            if not self.moire_pool:
                raise ValueError(
                    f"balanced=True requires at least one real moire sample, got {len(self.moire_pool)}; "
                    "there is no clean-source synthesis fallback"
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

    def _augment_sample(self, image: Image.Image, label: int, rng: np.random.Generator) -> Image.Image:
        """Shared class-aware augmentation path; the label is never changed.

        Clean fetches (label 0) only receive group augmentation when enabled —
        sample_moire_type and MOIRE_GENERATORS are unreachable here. Moire
        fetches (label 1) may gain one synthesized moire pattern (drawn with
        probability moire_augment_prob) before group augmentation.
        """
        if label == CLEAN_LABEL:
            if self.group_augment:
                return self._group_only(image=image, rng=rng)
            return image
        if self.moire_augment_prob > 0.0 and rng.random() < self.moire_augment_prob:
            type_name = sample_moire_type(rng=rng, type_names=self.type_names)
            generator_name = str(rng.choice(a=generator_names_for_type(type_name=type_name)))
            array = np.asarray(image, dtype=np.float32) / PIXEL_VALUE_MAX
            augmented, _params = MOIRE_GENERATORS[generator_name](image=array, rng=rng)
            if self.group_augment:
                augmented = self._apply_groups(image=augmented, rng=rng)
            return self._to_pil(array=augmented)
        if self.group_augment:
            return self._group_only(image=image, rng=rng)
        return image

    def __len__(self) -> int:
        if self.balanced:
            return 2 * self.num_slots
        return len(self.samples)

    def _getitem_balanced(self, index: int) -> tuple[Any, int]:
        """Serve the interleaved index space over the real class pools.

        Even indices return a clean fetch (label 0), odd indices a moire fetch
        (label 1); both pools cycle modulo their size, the smaller pool is
        oversampled, and every fetch runs the shared _augment_sample path.
        There is no synthesis fallback — the moire half always draws from the
        real moire pool.
        """
        slot, want_moire = divmod(index, 2)
        pool = self.moire_pool if want_moire else self.clean_pool
        label = MOIRE_LABEL if want_moire else CLEAN_LABEL
        source = pool[slot % len(pool)]
        image = Image.open(source["path"]).convert(mode=IMAGE_COLOR_MODE)
        image = self._augment_sample(image=image, label=label, rng=self._rng())
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __getitem__(self, index: int) -> tuple[Any, int]:
        if self.balanced:
            return self._getitem_balanced(index=index)
        sample = self.samples[index]
        image = Image.open(sample["path"]).convert(mode=IMAGE_COLOR_MODE)
        label = int(sample["label"])
        image = self._augment_sample(image=image, label=label, rng=self._rng())
        if self.transform is not None:
            image = self.transform(image)
        return image, label
