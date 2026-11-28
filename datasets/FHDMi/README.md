# FHDMi — local copy

Local copy of the FHDMi (Full High Definition Demoireing) dataset
(He, Wang, Shi & Duan, ECCV 2020, "FHDe2Net: Full High Definition Demoireing
Network", DOI 10.1007/978-3-030-58542-6_43). Downloaded 2026-11-27.

## What the dataset is

Real-captured 1080p (1920 × 1080) moiré/clean image pairs of screen-displayed
content; the first public demoiréing dataset at FHD resolution. Pairs are PNG
files with identical names in a `source/` (moiré input) and `target/`
(clean ground truth) folder.

- Official split: `train` (~10,000 pairs) and `test` (~2,500 pairs) — exact
  counts verified below after extraction.
- Used as a benchmark by ESDNet / UHDM / UnDeM and others.

## What is here

Downloaded from the official Google Drive folder `FHDMi_complete`
(https://drive.google.com/drive/folders/1IJSeBXepXFpNAvL5OyZ2Y1yu4KPvDxN5,
linked from the official code repo https://github.com/PKU-IMRE/FHDe2Net)
with anonymous `gdown`. A Baidu Pan mirror also exists
(https://pan.baidu.com/s/19LTN7unSBAftSpNVs8x9ZQ, code `jf2d`, login required).

The Drive folder stores each side of each split as **split chunks of one
tar.gz** (not standalone archives):

| Group | Files | Note |
|---|---|---|
| `train/source.tar.gz00–12` | 13 chunks | moiré training inputs |
| `train/target.tar.gz00–10` | 11 chunks | clean training targets |
| `test/source.tar.gz00–02` | 3 chunks | moiré test inputs |
| `test/target.tar.gz00–02` | 3 chunks | clean test targets |

Each chunk is ~1.15 GiB. Extraction requires all chunks of a group:

```
cd FHDMi_complete/train
cat source.tar.gz* | tar xz     # PowerShell: use cmd /c type ... | tar -xzf - -C .
```

Chunks are stored under `FHDMi_complete/{train,test}/` and pruned after the
corresponding group is fully extracted and verified.

## Caveats

- Google Drive enforces per-file anonymous download quotas; several chunks
  were quota-blocked at first attempt ("file has had many accesses" — resets
  within ~24 h). A retry daemon kept re-attempting the missing chunks every
  15 minutes until they succeeded; the status of every chunk is recorded in
  `docs/demoireing_datasets.html`.
- The FHDe2Net repo also ships `FHDMi_thin.txt` (a curated subset list for
  faster training); not needed here.

## Download tooling

`datasets/_tools/` holds the exact machinery used (gitignored like the data):
`fhdmi_daemon.py` + `daemon_tasks.json` (quota-retry loop, one gdown attempt
per missing chunk every 15 min) and `extract_fhdmi.py` (streams the chunks of
a complete group straight into tar — no combined archive on disk — then prunes
the chunks; `--list` prints the tar's internal paths). `watcher.py` invokes the
extractors automatically every 10 min and logs pool counts to
`_tools/watcher.log`.
