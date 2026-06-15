# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Extracts structured information from **mechanical engineering drawings** using Qwen2.5-VL-7B,
with an optional YOLO-OBB detection front-end. Two extraction targets:

1. **Title-block 5 fields** — `Title`, `Drawing No.`, `LIC. Material`, `Material`, `Rev` → strict JSON.
2. **Annotations** — `measure` (dimensions+tolerance), `gdt` (geometric tolerance frames), `radii` → strict JSON.

The guiding architecture for both is **"detect (where) → crop → VLM (what does it say)"**: a
detector/code decides *location*, the VLM only *reads text*. This role split is what reduces
hallucination. See `resource/DRAWING_EXTRACTION_GUIDE.md` for the full design rationale (in Korean).

## Environment

Everything runs in the conda env **`kardi_env`** (Python 3.10). Always use its interpreter
explicitly; do not assume the default `python` is correct:

```bash
/home/jhkim/anaconda3/envs/kardi_env/bin/python
```

`torch` here is a custom build (2.11+cu130). When installing packages that depend on torch
(notably `ultralytics`), use `--no-deps` to avoid clobbering it:

```bash
/home/jhkim/anaconda3/envs/kardi_env/bin/pip install ultralytics --no-deps
/home/jhkim/anaconda3/envs/kardi_env/bin/pip install opencv-python pyyaml pandas requests matplotlib psutil py-cpuinfo
```

## Common commands

**Title-block extraction (script):**
```bash
/home/jhkim/anaconda3/envs/kardi_env/bin/python extract_title_block.py [image_path]
# image_path defaults to test_title.png; writes <stem>_title_block.json next to the script
```

**Notebooks** (kernel = `kardi_env`) — these are the primary/most up-to-date workflow:
```bash
/home/jhkim/anaconda3/envs/kardi_env/bin/python -m jupyter nbconvert --to notebook --execute \
  --allow-errors --ExecutePreprocessor.timeout=540 --output /tmp/_out.ipynb extract_title_block.ipynb
```

**YOLO-OBB training / val / inference:**
```bash
/home/jhkim/anaconda3/envs/kardi_env/bin/python train_obb.py                          # train
/home/jhkim/anaconda3/envs/kardi_env/bin/python train_obb.py --model yolo11s-obb.pt --epochs 150
/home/jhkim/anaconda3/envs/kardi_env/bin/python train_obb.py --mode val
/home/jhkim/anaconda3/envs/kardi_env/bin/python train_obb.py --mode predict --source <drawing.jpg>
```
Trained weights land at `yolo_obb_drawing/<name>/weights/best.pt`; point the notebook's
section-7 `OBB_WEIGHTS` at that path to run the full annotation pipeline.

## Code map

| File | Role |
|---|---|
| `extract_title_block.py` | Standalone title-block extractor. Self-contained: prompt, `parse_json()`, `fix_material_case()`, inference. |
| `extract_title_block.ipynb` | **Main notebook.** §1–5 title-block (preprocessed: ROI crop+zoom, Rev/LIC special reads), §6 YOLO single-`cell` + rule mapping, §7 YOLO-OBB annotation pipeline. |
| `extract_title_block_hallusination.ipynb` | **Baseline** (no preprocessing — feeds full image). Exists only to demonstrate hallucinations the preprocessing fixes. §5b auto-compares the two JSON sets (model-free). |
| `train_obb.py` | YOLO-OBB train/val/predict. Drawing-specific aug baked in. |
| `drawing_obb.yaml` | OBB dataset config. |
| `resource/DRAWING_EXTRACTION_GUIDE.md` | Authoritative design doc (KR). |
| `resource/COMPARISON_RESULTS.md` | Preprocessed-vs-baseline accuracy comparison. |

Dirs: `input_doc/` (test images), `output/` (JSON results — `*_title_block.json` = preprocessed,
`*_title_block_baseline.json` = baseline), `datasets/drawing_obb/{images,labels}/{train,val}/`.

## Conventions and invariants that span files

- **Decoding is greedy** (`do_sample=False`, `torch.no_grad()`) → deterministic; re-running gives identical output.
- **Class index order must match everywhere:** `drawing_obb.yaml` `names` (`0=measure 1=gdt 2=radii`)
  ↔ notebook §7 `CLASS_NAMES`. Mismatch silently swaps labels.
- **VLM output is verbatim.** `parse_json()` extracts the first `{...}` (strips code fences).
  `fix_material_case()` corrects **letter case only** for known codes (e.g. `SCR18N8`→`SCr18N8`,
  see `CANONICAL_MATERIALS`) — it never adds/removes characters. Missing field → `null`.
- **European decimal commas** (`±0,1`, `+0,3`) are preserved in the `raw` field — never normalized to `.`.
- **OBB labels** are `class x1 y1 x2 y2 x3 y3 x4 y4`, normalized 0–1; `images/<split>/X.jpg` ↔ `labels/<split>/X.txt`.
- **Title-block detection uses a single `cell` class, not 5 field classes** — cells look alike, so
  YOLO detects boxes and *code* assigns fields by row/column position + label anchors (e.g. "LIC." text).

## Traps (from the guide — read before touching the OBB pipeline)

- `resource/doc_sample_paper*.jpg` is an **explanatory diagram**, NOT training data — it already has
  boxes/legend/3D model drawn on it. Label box-free original drawings instead.
- `train_obb.py` and `drawing_obb.yaml` contain **hardcoded absolute paths** to this project dir.
- `deskew_crop` in §7 has TODOs: rotation **sign** (flip `deg`→`-deg` if text comes out upside-down)
  and vertical-text 90° direction may need adjustment per dataset.
- `fliplr`/`flipud`/`hsv_*` augmentation is deliberately disabled — R/digits/symbols carry directional
  and the drawings are monochrome line art. Don't re-enable them.
- macOS paths (`/Users/...`) from elsewhere won't resolve here; copy files into the project dir.
