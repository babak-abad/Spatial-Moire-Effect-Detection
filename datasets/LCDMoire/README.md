# LCDMoire — local copy

Partial local copy of the LCDMoire demoiréing dataset (AIM 2019 challenge,
arXiv 1911.02498). Downloaded 2026-11-25 from a login-free community mirror
because every official channel is dead or gated (details below).

## What is here

`my_dataset_final.zip` (1,181,100,647 bytes), extracted to `my_dataset_final/`:

| Folder | Pairs | Format |
|---|---|---|
| `train/moire` + `train/gt` | 3,000 | JPEG 1024 × 1024 |
| `test/moire` + `test/gt` | 160 | JPEG 1024 × 1024 |

- This is a **subset** of the official 10,200-pair dataset (3,000 of the
  10,000 training pairs plus 160 evaluation images), JPEG-recompressed.
- Original LCDMoire images are PNG; this repack trades fidelity for size.
- Sample images were verified as genuine 1024 × 1024 JPEG moiré/clean pairs
  before downloading (dimensions and pair structure match the LCDMoire
  specification in the dataset paper).

Source (login-free): Google Drive file `my_dataset_final.zip`, linked from the
repo https://github.com/markxuzihe/moire-photo-detection-and-restoration
(its demoireing part uses MBCNN, the AIM 2019 challenge-winning network).

## Where the full dataset lives (status 2026-11)

- Official ETH links (`data.vision.ee.ethz.ch/timofter/AIM19demoire/`) —
  **dead** (HTTP 404; the Wayback Machine only ever captured a 404).
- CodaLab challenge (competitions 20165 / 20166) — registration required;
  the training data was distributed to registered participants only.
- Baidu Pan mirror (verified alive, full copy, extraction code `1234`):
  https://pan.baidu.com/s/1Nya-Hd_nfIa8WLfzs44gKQ
  - `Training.zip` — 4,271,680,478 bytes (10,000 training pairs)
  - `ValidationClear.zip` — 40,354,380 bytes (100 clean images)
  - `ValidationMoire.zip` — 144,684,259 bytes (100 moiré images)
  Downloading requires a Baidu account (and, for the 4 GB file, the Baidu
  Netdisk client).
- Test-set ground truth was never released (kept private by the challenge
  evaluation server).

## Usage

Pairs are `{split}/moire/*.jpg` (input) and `{split}/gt/*.jpg` (ground truth).
