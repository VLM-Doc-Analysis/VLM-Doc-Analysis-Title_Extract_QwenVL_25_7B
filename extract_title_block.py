"""Extract title-block fields from an engineering drawing using Qwen2.5-VL-7B.

Fields extracted: Title, Drawing No., LIC. Material, Material, Rev -> strict JSON.

Usage:
    python extract_title_block.py [image_path]
        image_path defaults to "input_doc/test_title_01.png"
"""

import json
import os
import re
import sys

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_IMAGE = os.path.join(SCRIPT_DIR, "input_doc", "test_title_01.png")

PROMPT = (
    "This image is the title block of a mechanical engineering drawing.\n"
    "Extract exactly the following five fields and return them as strict JSON:\n"
    '  "Title"         - the part/drawing title (the large text in the "Title" cell)\n'
    '  "Drawing No."   - the drawing number (the "Drawing No." cell)\n'
    '  "LIC. Material" - the value in the "LIC. Material / Blank" cell\n'
    '  "Material"      - the value in the "Material / Blank" cell '
    '(the one WITHOUT the "LIC." prefix)\n'
    '  "Rev"           - the revision value at the BOTTOM of the narrow far-right '
    '"Rev" column (a single digit/letter). Do NOT use the separate "Security" '
    'cell value (e.g. "B") for this field.\n'
    "Rules:\n"
    "- Use these exact JSON keys: \"Title\", \"Drawing No.\", "
    "\"LIC. Material\", \"Material\", \"Rev\".\n"
    "- Copy the text verbatim as printed on the drawing.\n"
    "- If a field cannot be found, set its value to null.\n"
    "- Output ONLY the JSON object. No markdown fences, no explanation."
)

# Canonical spellings (exact letter case) for known material codes. A model OCR
# value matching one of these case-insensitively is rewritten to the canonical
# casing. This corrects ONLY letter case -- it never changes which characters are
# present. Unknown values pass through unchanged. Add new codes as needed.
CANONICAL_MATERIALS = [
    "SCr18N8",
    "SUS316L",
    "SCrNi",
]


def fix_material_case(value):
    """Rewrite a known material code to its canonical letter case (case only)."""
    if not isinstance(value, str):
        return value
    key = value.strip().casefold()
    for canon in CANONICAL_MATERIALS:
        if canon.casefold() == key:
            return canon
    return value


def parse_json(text: str):
    """Pull the first {...} JSON object out of the model output."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not os.path.isfile(image_path):
        sys.exit(f"Image not found: {image_path}")

    stem = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_json = os.path.join(output_dir, f"{stem}_title_block.json")

    print(f"Loading {MODEL_ID} ...", file=sys.stderr)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    print("Running inference ...", file=sys.stderr)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    try:
        result = parse_json(raw)
    except json.JSONDecodeError as exc:
        print(f"Failed to parse JSON ({exc}). Raw model output:\n{raw}", file=sys.stderr)
        sys.exit(1)

    # Fix letter case of known material codes (e.g. "SCR18N8" -> "SCr18N8").
    for field in ("LIC. Material", "Material"):
        if field in result:
            fixed = fix_material_case(result[field])
            if fixed != result[field]:
                print(f'{field} case-fixed: "{result[field]}" -> "{fixed}"', file=sys.stderr)
                result[field] = fixed

    pretty = json.dumps(result, ensure_ascii=False, indent=2)
    print(pretty)
    with open(output_json, "w", encoding="utf-8") as fh:
        fh.write(pretty + "\n")
    print(f"\nSaved -> {output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
