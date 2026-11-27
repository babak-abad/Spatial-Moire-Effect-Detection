# Augmentation pipeline — moiré synthesis plus capture-style group corruption

This document is the canonical description of how augmented data is produced: a
two-stage pipeline that turns a clean screen photograph into a labeled moiré
sample. **Stage 1** synthesizes the moiré pattern itself (procedural generators
driven by the weighted type registry of `docs/moire_types.md`). **Stage 2**
optionally corrupts the result — and, through the same decision, real moiré
photographs and clean images that stay clean, both of which skip Stage 1 —
with a random subset of capture-style augmentation groups (imgaug) that
emulate sensor read noise, exposure error, tone-curve drift, defocus/shake
blur, dead pixels and camera-angle drift — so moiré evidence, synthetic or
real, is degraded the way a fresh capture would degrade it while the clean
class sees the very same corruption, never letting degradation itself
become a moiré cue.

Machine-readable counterparts:

| Concern | Module |
|---|---|
| Weighted moiré type registry | `src/spatial_moire/moire_types.py` (`MOIRE_TYPES`) |
| Moiré pattern generators | `src/spatial_moire/synthesis/augment.py` (`MOIRE_GENERATORS`) |
| Augmentation group registry + algorithm | `src/spatial_moire/synthesis/groups.py` (`AUGMENTATION_GROUPS`, `apply_group_augmentation`) |
| Every hard-coded range | `src/config.py` (`GROUP_*`, per-generator `*_RANGE`) |
| On-the-fly training dataset | `src/spatial_moire/baseline/augmentation.py` (`augmented_moire_dataset`) |
| Balanced train batches | `src/spatial_moire/baseline/data.py` (`balanced_moire_dataset`, `balanced_shuffle_sampler`), `src/config.py` (`TRAIN_BALANCED_BATCHES`) |
| Offline sample generation | `src/generate_moire_samples.py` |
| Dataset source pools & split fractions | `src/config.py` (`CLEAN_IMAGE_FOLDERS`, `MOIRE_IMAGE_FOLDERS`, `TRAIN_FRACTION` / `VAL_FRACTION` / `TEST_FRACTION`) — see `docs/dataset.md` |

## Pipeline overview

```mermaid
flowchart TD
    A["Clean source image<br/>(HxWx3 uint8, label 0)"] --> B{"Augment draw:<br/>rng.random() < augment_prob ?"}
    B -- "yes · label 1" --> C["Stage 1 · draw moiré type<br/>possibility = weight / total_weight"]
    C --> D["Stage 1 · draw generator<br/>uniform among the type's mapped generators"]
    D --> E["Stage 1 · moiré synthesis<br/>generator(image, rng) → image + params"]
    E --> F{"group_augment<br/>enabled?"}
    B -- "no · label 0" --> F
    F -- "no" --> H["Clip to [0, 1] · scale to uint8"]
    F -- "yes" --> G["Stage 2 · apply_group_augmentation<br/>random ordered subset of the six groups<br/>(each step clipped to [0, 1])"]
    G --> H
    H -- "label 1" --> I["Moire sample<br/>(label 1)"]
    H -- "label 0" --> Z["Clean sample<br/>(label 0)"]
    M["Real moiré source image<br/>(moiré innate to the capture, label 1)"] --> P["Pass through unchanged<br/>(no moiré synthesis, label 1)"]
    P --> F
    style I fill:#1b5e20,stroke:#1b5e20,color:#ffffff
    style Z fill:#0d47a1,stroke:#0d47a1,color:#ffffff
    style M fill:#4a148c,stroke:#4a148c,color:#ffffff
    style P fill:#4a148c,stroke:#4a148c,color:#ffffff
```

The label is settled before Stage 2: the augment draw alone flips a clean
image to label 1, and group corruption never changes it. Real moiré images
— photographs that innately show the effect, label 1 — never enter Stage 1:
they pass through moiré synthesis unchanged and then flow into the same
`group_augment enabled?` decision as the synthetic path. Clean images that
lose the augment draw join that decision as well. When group augmentation
is disabled (or draws zero groups) the clip/scale round-trip is lossless,
so both reach the output pixel-identical to the source file with their
original label; when it is enabled they receive the same capture-style
corruption as the synthetic positives — the real moiré evidence being
degraded the way a fresh capture would degrade it, and the clean image
staying clean under the very corruptions the positives see, so degradation
itself can never act as a moiré cue. Either way every sample undergoes only
the standard train transform (crop/flip/tensor). Source files are never
modified on any path.

## Stage 1 — moiré synthesis

1. **Type draw** — `sample_moire_type(rng=...)` draws one of the nine registry
   types with possibility `weight / total_weight` (see
   `docs/moire_types.md` for the taxonomy, weights and per-type causes).
2. **Generator draw** — the type maps to one or two generator names
   (`generator_names_for_type`); one is drawn uniformly.
3. **Synthesis** — the generator returns `(image, params)`. Most generators
   blend a procedural pattern via
   `image * (1 + strength * (2 * pattern - 1))`, clipped to `[0, 1]`;
   `screen_shoot` mixes a display-grid/camera-grid simulation by `strength`;
   `nested_moire` applies two other generators in sequence. All random
   parameters come from the caller's numpy `Generator`.

## Stage 2 — group augmentation algorithm

`apply_group_augmentation(image=..., rng=..., group_subset=...,
num_groups_range=...)` in `src/spatial_moire/synthesis/groups.py`:

```text
validate image shape HxWx3 and dtype float32      → ValueError otherwise
                                                   (values in [0, 1] are the caller's contract)
(low, high) = validated num_groups_range           (default (2, 4))
selected    = validated group names                (all six by default)

num_groups = min(rng.integers(low, high + 1), len(selected))
chosen     = rng.choice(selected, size=num_groups, replace=False)   # random order

result = image
for group_name in chosen:                          # apply in drawn order
    factory = factories[rng.integers(0, len(factories))]  # one augmenter per group
    seed    = rng.integers(0, 2**31)                      # child augmenter seed
    result  = factory(seed).augment_image(result)
    result  = clip(result, 0, 1).astype(float32)          # clip after EVERY step

return result, {count_range: (low, high), num_groups,
                groups: [{group, augmenter, seed}, ...]}
```

Key properties:

- **One augmenter per group** — a group holds several alternative imgaug
  factories; exactly one is drawn per applied group.
- **Subset without replacement, random order** — a group appears at most once
  per image and the application order varies across draws.
- **Per-step clipping** — imgaug targets uint8 while this pipeline works on
  float32 `[0, 1]`; some augmenters (Multiply, additive noise, blur edges)
  overflow the range, so each step is clipped back before the next.
- **Determinism** — the subset size, the subset itself, the order, the factory
  pick and every child seed are all drawn from the *caller's* `rng`, so an
  identical seed reproduces a bit-identical image *and* recipe; the recipe is
  JSON-friendly and is recorded in synthesis manifests.

## Group registry

| Group | Physical motivation | imgaug augmenters (one drawn) | Key ranges (config) |
|---|---|---|---|
| `noise` | sensor read noise | `AdditiveGaussianNoise` (×2, one per-channel), `AdditiveLaplaceNoise` | σ 0.01–0.05, scale 0.01–0.04 |
| `photometric` | exposure / gain error | `Multiply` (×2, one per-channel 50%), `AddElementwise` | mul 0.7–1.3, add ±0.08 |
| `contrast` | tone-curve drift | `GammaContrast`, `LinearContrast`, `SigmoidContrast` | γ 0.7–1.4, α 0.75–1.3, gain 3–10 / cutoff 0.4–0.6 |
| `blur` | defocus / camera shake | `GaussianBlur`, `AverageBlur`, `MotionBlur` | σ 0–1.2, k 3–5, k=7 / angle ±45° |
| `dropouts` | dead pixels / lost packets | `CoarseDropout`, `Dropout` (per-channel 50%) | p 0–0.05 / size 0.1–0.3, p 0–0.02 |
| `geometric` | camera-angle drift | `Affine` (scale+rotate), `Affine` (shear+translate); bilinear, constant fill 0.5, skimage backend | scale 0.92–1.08, rotate ±8°, shear ±6°, translate 0–4% |

Default applied count per image: `GROUP_AUGMENT_COUNT_RANGE = (2, 4)`.

## The two consumers

### On-the-fly training augmentation

`augmented_moire_dataset` (in `src/spatial_moire/baseline/augmentation.py`),
used by `src/train_baseline.py`. Its samples come from the config-driven
pooled split (see `docs/dataset.md`): training feeds it the train portion of
`split_train_val_test` and validates on the val portion:

```mermaid
flowchart TD
    A["Image.open + convert"] --> L{"label ?"}
    L -- "clean (0)" --> B{"moire draw:<br/>rng.random() < augment_prob ?"}
    B -- "yes · label 1" --> S["sample_moire_type → draw generator →<br/>MOIRE_GENERATORS"]
    B -- "no · label 0" --> F{"group_augment<br/>enabled?"}
    L -- "real moiré (1):<br/>no moiré synthesis" --> F
    S --> F
    F -- "no" --> U["clip, scale to uint8, PIL image<br/>(label settled before the groups)"]
    F -- "yes" --> G["apply_group_augmentation<br/>from the worker rng"]
    G --> U
    U --> T["train transform (crop/flip/tensor)"]
    T --> O["(image, label) to the DataLoader"]
```

- Each DataLoader worker owns its own `Generator` seeded `seed + worker_id`,
  so augmentation is reproducible and workers never share state.
- Constructor options: `group_augment` (off by default), `group_names`
  (subset, all by default), `num_groups_range` (default `(2, 4)`) — invalid
  names or ranges raise `ValueError` at construction time.
- The moiré-augmented share is controlled by `augment_prob`
  (`--moire-augment-prob`); the label follows the moiré draw alone — a
  moiré-augmented image always carries the moiré label, and a
  group-corrupted clean image stays clean, so corruption itself can never
  leak the label.
- Real moiré images are decided first and skip moiré synthesis: they flow
  straight into the group_augment decision, so with `group_augment=True`
  a genuine positive receives the same capture-style groups (drawn from
  the worker rng) as a synthetic one, and with it off reaches the
  transform exactly as loaded from disk — pinned by
  `check_real_moire_images_receive_group_augmentation` and
  `check_real_moire_group_path_matches_pipeline`.
- Clean images that lose the moiré draw enter the same decision: with
  `group_augment=True` they receive the same capture-style groups while
  keeping label 0, and with it off reach the transform exactly as loaded
  from disk — pinned by `check_clean_images_receive_group_augmentation`
  and `check_clean_group_path_matches_pipeline`. When the moiré draw is
  impossible (`augment_prob = 0` or an empty type selection) it is skipped
  without consuming rng, so those images go straight to the group draw
  exactly like real moiré.

#### Balanced train batches (default)

`TRAIN_BALANCED_BATCHES = True` makes `src/train_baseline.py` serve exactly
balanced train batches: every batch of an even `batch_size` holds
`batch_size // 2` clean and `batch_size // 2` moire items. The mechanism is the
interleaved virtual index space of `balanced_moire_dataset` — even indices
address the clean pool, odd indices the moire pool, both cycling modulo their
pool size — combined with `balanced_shuffle_sampler`, which shuffles the slot
order (seeded, reshuffled every epoch) while keeping each (clean, moire) pair
adjacent, so no contiguous window of the stream can drift off balance. The
smaller class pool is oversampled to `max(n_clean, n_moire)` slots per class
per epoch; an odd `batch_size` (or below 2) is rejected up front, and
`--no-balance-batches` restores plain shuffled batches.

Inside `augmented_moire_dataset(balanced=True)` the parity also fixes the
label: clean slots never draw moiré synthesis (they may still receive the
capture-style groups), and moire slots are a real moiré source passed through
or — with probability `augment_prob` — a clean source with forced moiré
synthesis, so `augment_prob` becomes the synthesized share of the moire half.
With no moiré sources at all, balance is kept through synthesis alone
(`augment_prob > 0` then required); an empty clean pool raises `ValueError`.

### Offline sample generation

`src/generate_moire_samples.py` writes JPEG samples, previews and
`moire_synthesis_params.json`. It only ever consumes **clean** source
images — its job is manufacturing synthetic positives, so real moiré
images never flow through it:

- `--sampling grid` renders one sample per selected type per source image
  (full preview grid); `--sampling weighted` draws each sample's type by
  `weight / total_weight`.
- Both modes route through `render_moire_type`, which runs the mapped
  generator and then, with `--groups`, the group augmentation, merging the
  recipe into the entry's `params["groups"]`.
- The manifest's top-level `group_augmentation` block records
  `{group_names, count_range}` (or `null` when off) alongside `seed`,
  `sampling`, `type_probabilities` and per-entry `type` / `generator` /
  `variant` / `probability` / `params`.

## CLI flags

| Script | Flag | Effect |
|---|---|---|
| `src/generate_moire_samples.py` | `--groups` | corrupt every sample with the group pipeline |
| | `--group-names NAME ...` | restrict the draw to a subset of groups (all by default) |
| | `--group-count LOW HIGH` | inclusive applied-groups range (default `(2, 4)`) |
| `src/train_baseline.py` | `--group-augment` | corrupt every training image with the groups, labels unchanged |
| | `--augment-groups NAME ...` | restrict the group subset (all by default) |
| | `--augment-group-count LOW HIGH` | inclusive applied-groups range (default `(2, 4)`) |
| | `--no-balance-batches` | fall back from exactly half/half train batches (default) to plain shuffled batches |

Group augmentation composes with the existing moiré flags
(`--moire-augment-prob`, `--moire-augment-types`, `--types`, `--sampling`,
`--variants`, `--seed`) and with the balanced-batch default
(`--no-balance-batches` opts out).

## Validation guarantees

`tests/test_groups.py` pins the contract:

- every factory builds a named augmenter; every group preserves shape, float32
  dtype and finiteness on `[0, 1]` input;
- pipeline output stays HxWx3 float32 in `[0, 1]`; fixed counts apply exactly
  that many unique groups; requested subsets are honored; order varies;
- a `(0, 0)` count is an exact identity;
- identical seeds reproduce images and recipes, different seeds differ;
- invalid images, count ranges and group names raise `ValueError`;
- groups compose after moiré synthesis; the dataset relabels clean images,
  reproduces across equal seeds, changes pixels relative to plain moiré
  augmentation, and group-augments both real moiré and clean images that
  stay clean reproducibly when enabled, each keeping its label
  (pixel-identical otherwise, straight into the group draw);
- group-augmented images flow through the train transform as
  `(3, crop, crop)` tensors.

`tests/test_balanced_batches.py` pins the balanced-batch contract: parity
labels and the `2 * max(pool)` length, exact half/half batches through a real
DataLoader (also with workers and unbalanced sources), `augment_prob`
semantics per parity, sampler adjacency / reproducibility / per-epoch
reshuffle, `ValueError` on empty pools, and unchanged legacy behavior when
`balanced=False`.

## Tuning

1. Edit the `GROUP_*` values in `src/config.py` (ranges are inclusive
   `(low, high)` tuples consumed by the imgaug factories).
2. Change the per-image count via `GROUP_AUGMENT_COUNT_RANGE` or the CLI
   `--group-count` / `--augment-group-count` overrides.
3. Add a group by appending a name → tuple-of-factories entry to
   `AUGMENTATION_GROUPS` (plus new config ranges); unknown names and empty
   selections are rejected, and `tests/test_groups.py` expects the registry to
   stay populated (≥ 5 groups, ≥ 1 factory each).

## Scope and compatibility notes

- **imgaug 0.4.0 predates numpy 2.x**: `groups.py` patches the removed
  `np.sctypes` / `np.bool` / `np.complex` aliases *before* importing imgaug
  (imgaug reads them at import time). The patch is guarded by `hasattr` so it
  stays inert on future imgaug releases.
- **Salt / Pepper augmenters are excluded** on purpose: they write raw 0–255
  values on float arrays, so pepper would turn into white specks after the
  `[0, 1]` clipping. `CoarseDropout` / `Dropout` cover the dead-pixel look.
- **Groups run after moiré synthesis whenever synthesis happens, never
  before**: the moiré pattern itself must be degraded the way real capture
  degrades it, not bypassed. Real moiré inputs and clean inputs that stay
  clean skip synthesis but still pass through the group decision
  afterwards, with their labels unchanged.
- Group augmentation never writes to source files; only in-memory copies
  (training) or new output files (sample generation) are affected. Offline
  sample generation still consumes clean sources only — the real-moiré and
  stay-clean branches of the flow above exist in the training dataset
  alone.