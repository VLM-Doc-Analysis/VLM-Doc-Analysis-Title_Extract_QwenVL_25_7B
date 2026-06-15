# Title Extraction from Engineering Drawings — Qwen2.5-VL-7B

Structured information extraction from **mechanical engineering drawings** using
[Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), with an
optional YOLO-OBB detection front-end.

Two extraction targets:

1. **Title-block — 5 fields** → `Title`, `Drawing No.`, `LIC. Material`, `Material`, `Rev` (strict JSON)
2. **Annotations** → `measure` (dimensions + tolerance), `gdt` (geometric tolerance frames), `radii` (strict JSON)

## Core idea: *detect (where) → crop → VLM (what does it say)*

Both pipelines split responsibility to reduce hallucination:

| Concern | Owner |
|---|---|
| **Where** is the cell / annotation? | detector (YOLO) or fixed-ROI code |
| **Which field** is it? | code (row/column geometry + label anchors) |
| **What text** does it contain? | the VLM (Qwen2.5-VL) reads the crop |

Letting code (not the network) decide *location* is what keeps `Rev` from being confused with
`Security`, prevents an empty cell from being filled with a neighbor's value, and keeps small
text (revision tables, material codes) readable. See
[`resource/DRAWING_EXTRACTION_GUIDE.md`](resource/DRAWING_EXTRACTION_GUIDE.md) for the full
design rationale (Korean).

## Why preprocessing matters

The repo ships two notebooks that differ **only** in preprocessing, so the effect is measurable:

- `extract_title_block.ipynb` — preprocessed (ROI crop + zoom, dedicated Rev/LIC reads)
- `extract_title_block_hallusination.ipynb` — baseline (full image straight into the model)

On 3 test drawings, preprocessing corrected **3 hallucinations** the baseline produced (lost
spaces in titles, an empty cell copied from its neighbor, a 7-row revision table collapsed to 1).
Full table: [`resource/COMPARISON_RESULTS.md`](resource/COMPARISON_RESULTS.md). Decoding is
greedy (`do_sample=False`), so results are deterministic and reproducible.

## Repository layout

```
extract_title_block.py        # Standalone title-block extractor (script form)
extract_title_block.ipynb     # Main notebook: §1–5 title-block, §6 YOLO cell-mapping, §7 OBB pipeline
extract_title_block_hallusination.ipynb  # Baseline (no preprocessing) + §5b auto-comparison
train_obb.py                  # YOLO-OBB train / val / predict
drawing_obb.yaml              # OBB dataset config (0=measure 1=gdt 2=radii)
input_doc/                    # Test drawing images
output/                       # JSON results (*_title_block.json / *_baseline.json)
datasets/drawing_obb/         # YOLO-OBB images/ + labels/ (train|val)
resource/                     # Design guide + comparison results + sample diagram
```

## Setup

Built and tested with the `kardi_env` conda environment (Python 3.10).

> **⚠️ torch is a custom CUDA build (`2.11.0+cu130`).** Do **not** reinstall it from plain PyPI —
> it is intentionally omitted from `requirements.txt`.

```bash
pip install -r requirements.txt

# Optional — for the YOLO-OBB annotation pipeline only.
# Use --no-deps so ultralytics does not replace the custom torch build:
pip install ultralytics --no-deps
pip install opencv-python pyyaml pandas matplotlib psutil py-cpuinfo
```

## Usage

### Title-block extraction (script)

```bash
python extract_title_block.py [image_path]
# image_path defaults to input_doc/test_title_01.png
# writes output/<stem>_title_block.json
```

Output example:

```json
{
  "Title": "FLANGE, CIRCULAR PLAIN",
  "Drawing No.": "A14-640003-8",
  "LIC. Material": "SCr18N8",
  "Material": "SUS316L",
  "Rev": ["0"]
}
```

The script copies model text **verbatim**; `fix_material_case()` corrects only the *letter case*
of known material codes (e.g. `SCR18N8` → `SCr18N8`) and never alters characters. Missing
fields become `null`.

### Notebooks

Run top-to-bottom with the `kardi_env` kernel, or headless:

```bash
jupyter nbconvert --to notebook --execute --allow-errors \
  --ExecutePreprocessor.timeout=540 --output /tmp/_out.ipynb extract_title_block.ipynb
```

### YOLO-OBB annotation pipeline (optional)

```bash
python train_obb.py                                  # train
python train_obb.py --model yolo11s-obb.pt --epochs 150
python train_obb.py --mode val
python train_obb.py --mode predict --source <drawing.jpg>
```

Trained weights land in `yolo_obb_drawing/<name>/weights/best.pt`; point the notebook's §7
`OBB_WEIGHTS` at that path to run the full *detect → deskew-crop → VLM read* pipeline.

**OBB label format** (`labels/<split>/<name>.txt`, one box per line, normalized 0–1):

```
class  x1 y1 x2 y2 x3 y3 x4 y4      # class: 0=measure 1=gdt 2=radii
```

## Notes & gotchas

- **Class-index order must match** between `drawing_obb.yaml` `names` and the notebook §7
  `CLASS_NAMES` — a mismatch silently swaps labels.
- `resource/doc_sample_paper*.jpg` is an **explanatory diagram, not training data** — it already
  has boxes/legend drawn on it. Label box-free original drawings instead.
- European decimal commas (`±0,1`, `+0,3`) are preserved in the `raw` field — never normalized to `.`.
- Augmentation flips (`fliplr`/`flipud`) and color jitter are intentionally disabled: R/digits/
  symbols carry directional meaning and the drawings are monochrome line art.

Detailed Korean documentation lives in [`resource/`](resource/).
