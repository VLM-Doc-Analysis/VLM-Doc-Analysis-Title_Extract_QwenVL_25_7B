# Title Extraction from Engineering Drawings — Qwen2.5-VL-7B

Structured information extraction from **mechanical engineering drawings** using
[Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), with an
optional YOLO-OBB detection front-end.

Two extraction targets:

1. **Title-block — 5 fields** → `Title`, `Drawing No.`, `LIC. Material`, `Material`, `Rev` (strict JSON)
2. **Annotations** → `measure` (dimensions + tolerance), `gdt` (geometric tolerance frames), `radii` (strict JSON)

## Recommended approach (per literature + scope)

Based on a method comparison against two on-topic papers (Khan et al.) — see
[`resource/도면요소추출_방법비교.md`](resource/도면요소추출_방법비교.md):

- **Title-block (5 fixed fields)** → **Qwen2.5-VL zero-shot + ROI preprocessing** (already implemented;
  no training needed).
- **Annotations (measure / radii / gdt)** → **YOLO detect → fine-tuned Donut**, which is the
  literature SOTA for this task (lightweight ~143M, GD&T F1 ≈ 0.965, low hallucination).
- **Cold start (little/no labeled data)** → bootstrap with a large VLM (Qwen/PaddleOCR) to
  **auto-label crops**, accumulate Donut training data, then fine-tune. This repo's
  [`donut_data/`](donut_data/) pipeline does exactly that.

### Method comparison — key conclusions

Three approaches were weighed (Donut / PaddleOCR-VL / Qwen-VL); full analysis, per-element
matrix, and appendices are in [`resource/도면요소추출_방법비교.md`](resource/도면요소추출_방법비교.md).

| Element | Best choice | Why |
|---|---|---|
| Title-block (5 fields) | **Qwen zero-shot + ROI** | fixed schema; already works, no training |
| measure / radii / GD&T | **fine-tuned Donut** | papers: GD&T F1 ≈ 0.965, ~143M, low hallucination |
| Notes (free text) | Qwen / PaddleOCR | handwriting + open vocabulary |

- **Fine-tuned lightweight Donut beats large zero-shot VLMs** (GPT-4o/Claude/Qwen) for this
  domain and hallucinates less → it is the *production target*; large VLMs serve cold-start
  auto-labeling / fallback.
- **GD&T is a strength, not a weakness, for fine-tuned Donut**: the symbol set is finite
  (14, ASME Y14.5) and schema-constrained, so the "vocab" concern is a *zero-shot-only* issue
  that fine-tuning dissolves.
- **Fine-tuning reduces but does not eliminate hallucination**; the first line of defense is
  preprocessing (ROI crop) + strict-JSON/`null` discipline, not the model itself.

## Core idea: *detect (where) → crop → VLM (what does it say)*

Both pipelines split responsibility to reduce hallucination:

| Concern | Owner |
|---|---|
| **Where** is the cell / annotation? | detector (YOLO) or fixed-ROI code |
| **Which field** is it? | code (row/column geometry + label anchors) |
| **What text** does it contain? | a VLM reads the crop — Qwen2.5-VL (title-block) / fine-tuned Donut (annotations) |

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
train_obb.py                  # YOLO-OBB train / val / predict (annotations: measure/gdt/radii)
drawing_obb.yaml              # OBB dataset config (0=measure 1=gdt 2=radii)
train_layout.py               # YOLOv11-det train/val/predict (layout: title_block/view/note)
drawing_layout.yaml           # layout-det dataset config (0=title_block 1=view 2=note)
donut_data/                   # Donut training-data labeling pipeline (see donut_data/README.md)
  ├ build_dataset.py          #   detect/crop → Qwen auto-label → Donut format + manifest
  ├ autolabel.py · schemas.py #   Qwen labeler + per-class JSON schema/prompts
  ├ to_donut_vml.py           #   convert output → donut_vml training format + CFG
  └ donut_data_pipeline.ipynb #   runnable notebook version of the above
input_doc/                    # Test drawing images
output/                       # JSON results (*_title_block.json / *_baseline.json)
datasets/drawing_obb/         # YOLO-OBB images/ + labels/ (train|val)
resource/                     # Design guide, method comparison (도면요소추출_방법비교.md), samples
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

### Layout detector (YOLOv11-det: title_block / view / note)

Robust, format-independent alternative to fixed-ROI cropping (see comparison doc §14). Detects
the title-block / view / note regions so crops adapt across drawing formats.

```bash
python train_layout.py                               # train (data: datasets/drawing_layout/)
python train_layout.py --mode predict --source <drawing.png>
# use the trained detector to crop title blocks:
python donut_data/collect_titleblock.py --input <drawings> \
    --weights yolo_layout_drawing/train/weights/best.pt --out patches/titleblock --preview
```
Label format (det, normalized): `class cx cy w h` (`0=title_block 1=view 2=note`). Until trained,
`collect_titleblock.py` falls back to an orientation-based heuristic crop.
**How to label** both detectors (layout det + annotation OBB): see
[`resource/LABELING_GUIDE.md`](resource/LABELING_GUIDE.md).

### Donut training-data pipeline (label → fine-tune)

Build a Donut training set (image + JSON) from crops, with a large VLM auto-labeling the
bootstrap pass. Full docs: [`donut_data/README.md`](donut_data/README.md).

```bash
# 1) Cold start — auto-label pre-cut crops (patches/<class>/*.png), no YOLO needed
python donut_data/build_dataset.py --mode patches --input patches \
    --out datasets/donut_anno --classes measure radii gdt

# 2) (after review) review manifest.csv -> set verified=1, fix labels/*.json

# 3) Convert to donut_vml training format (+ prints a ready-to-paste CFG)
python donut_data/to_donut_vml.py --src datasets/donut_anno \
    --dvml /home/jhkim/projects/donut_vml --class gdt --verified-only
```

Then fine-tune in `donut_vml/donut_training.ipynb` (local mode, paste the printed CFG).
Generated crops/datasets are gitignored (derived from confidential drawings).

## Notes & gotchas

- **Class-index order must match** between `drawing_obb.yaml` `names` and the notebook §7
  `CLASS_NAMES` — a mismatch silently swaps labels.
- `resource/doc_sample_paper*.jpg` is an **explanatory diagram, not training data** — it already
  has boxes/legend drawn on it. Label box-free original drawings instead.
- European decimal commas (`±0,1`, `+0,3`) are preserved in the `raw` field — never normalized to `.`.
- Augmentation flips (`fliplr`/`flipud`) and color jitter are intentionally disabled: R/digits/
  symbols carry directional meaning and the drawings are monochrome line art.

Detailed Korean documentation lives in [`resource/`](resource/) — notably the design guide
([`DRAWING_EXTRACTION_GUIDE.md`](resource/DRAWING_EXTRACTION_GUIDE.md)) and the method comparison
& recommendation ([`도면요소추출_방법비교.md`](resource/도면요소추출_방법비교.md)).
