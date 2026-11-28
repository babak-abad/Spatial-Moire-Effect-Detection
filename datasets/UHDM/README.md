# UHDM — local copy

Local copy of the UHDM (Ultra-High-Definition Demoiréing) dataset
(Yu et al., ECCV 2022, "Towards Efficient and Scale-Robust Ultra-High-Definition
Image Demoiréing", arXiv 2207.09935 — the ESDNet paper). Downloaded 2026-11-27.

## What the dataset is

5,000 real-world 4K moiré/clean image pairs, photographed with smartphones on a
DJI OM 5 gimbal shooting display content (multiple phones × display panels,
varied camera views → wide moiré scale variation).

- Official split: 4,500 training / 500 testing pairs (random split).
- Resolution: native 4032 × 3024 up to 4624 × 3472; the published benchmark
  evaluates on center-cropped 3840 × 2160.
- Pair naming inside each split folder: `NNNN_moire.jpg` (degraded input) and
  `NNNN_gt.jpg` (clean ground truth), JPEG.

## What is here

Downloaded from the official Google Drive folder `UHDM_DATA`
(https://drive.google.com/drive/folders/1DyA84UqM7zf3CeoEBNmTi_dJ649x2e7e,
linked from the project page https://xinyu-andy.github.io/uhdm-page/ and the
official code repo https://github.com/CVMI-Lab/UHDM) with anonymous `gdown`.

Archives downloaded into `_archives/` (pruned after extraction):

| Archive | Size | Contents |
|---|---|---|
| `train.tar.gz` (as `train_split00–17.tar.gz`) | ~38.7 GiB (18 × ~2.15 GiB) | training pairs |
| `test.tar.gz` | 5.94 GiB | test pairs (center-cropped 4K) |

- The monolithic `train.tar.gz` from the repo's `scripts/download_data.sh`
  (file id `1NiTohc65F…`) was Google-Drive quota-blocked ("too many accesses"),
  so the equivalent `train_split` parts from the official folder were used.
- `test_origin.tar.gz` (file id `1SGVXv1Ce8…`, the original non-center-cropped
  test images) was also quota-blocked at download time; it is not needed for
  training-pool use (the cropped `test.tar.gz` covers the 500 test pairs).
  Status is tracked in `docs/demoireing_datasets.html`.

Extracted layout — reorganized for this repo's folder-based class pools
(the original tars keep moiré / ground-truth side by side with
`NNNN_moire.jpg` / `NNNN_gt.jpg` naming; per-device subfolders are folded
into a filename prefix so names stay unique and pair stems stay aligned):

| Folder | Contents |
|---|---|
| `train/moire/` | training moiré inputs (`[device__]NNNN_moire.jpg`) |
| `train/gt/` | training clean ground truths (same stems) |
| `test/moire/`, `test/gt/` | the 500 test pairs, same scheme |

## Usage

Moiré-class pool: `datasets/UHDM/{train,test}/moire`; clean-class pool:
`datasets/UHDM/{train,test}/gt` (wired into `src/config.py`).

## Caveats

- Drive distributes the train set as 18 standalone tar.gz parts
  (`train_splitNN.tar.gz`), each holding a subset of the training folders.
- Downloads were anonymous (~1–2 MB/s aggregate); full retrieval takes hours.
  See `docs/demoireing_datasets.html` for the final verification numbers.

## Download tooling

`datasets/_tools/` holds the exact machinery used (gitignored like the data):
`download_manager.py` + `tasks.json` (parallel gdown queue),
`extract_uhdm.py` (extract → reorganize into `train|test` × `moire|gt` →
prune) and `watcher.py` (runs the extractor every 10 min as archives land and
appends pool counts to `_tools/watcher.log`). All steps are idempotent and
safe to re-run.
