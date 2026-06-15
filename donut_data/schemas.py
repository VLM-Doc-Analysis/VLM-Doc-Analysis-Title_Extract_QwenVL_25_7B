"""Donut 학습용 라벨 스키마 · 클래스별 task_prompt · Qwen 자동라벨 프롬프트.

도면 요소를 Donut(이미지→JSON)으로 학습하기 위한 "정답 JSON" 형식을 한곳에 정의한다.
- 비교문서(resource/도면요소추출_방법비교.md) 0-A 권장: 주석(measure/gdt/radii)은 Donut,
  표제란 5필드는 Qwen zero-shot. 본 파이프라인은 둘 다 라벨링할 수 있게 스키마를 제공한다.
- 클래스 인덱스/이름은 drawing_obb.yaml(0=measure 1=gdt 2=radii)과 일치시킬 것.
"""

# Donut 디코더의 시작 토큰 겸 태스크 식별자. 학습 시 tokenizer 에 special token 으로 추가.
TASK_PROMPT = {
    "measure":    "<s_measure>",
    "radii":      "<s_radii>",
    "gdt":        "<s_gdt>",
    "titleblock": "<s_titleblock>",
}

# 클래스별 빈 스키마(=필드 목록). --no-model(수작업) 모드에서 템플릿으로 사용.
EMPTY_SCHEMA = {
    "measure":    {"nominal": None, "upper": None, "lower": None, "raw": None},
    "radii":      {"nominal": None, "upper": None, "lower": None, "raw": None},
    "gdt":        {"characteristic": None, "symbol": None, "tolerance": None,
                   "modifier": None, "datums": [], "raw": None},
    "titleblock": {"Title": None, "Drawing No.": None, "LIC. Material": None,
                   "Material": None, "Rev": None},
}

CLASSES = list(EMPTY_SCHEMA.keys())

# ── Qwen 자동라벨 프롬프트(부트스트랩용). 출력은 strict JSON 한 개. ──────────────
_MEASURE_RULES = (
    "Extract the dimension/tolerance into STRICT JSON with keys exactly:\n"
    '  "nominal" - the nominal value as printed (e.g. "Ø36 H8", "78", "20")\n'
    '  "upper"   - upper tolerance as printed (e.g. "+0,3", "+0,1") or null\n'
    '  "lower"   - lower tolerance as printed (e.g. "-0,1", "0") or null\n'
    '  "raw"     - verbatim transcription of the whole patch\n'
    "Rules: copy VERBATIM; preserve European decimal commas (\"0,1\" stays \"0,1\"); "
    "keep \"Ø\" in nominal; if a field is absent use null. Output ONLY the JSON object."
)

PROMPTS = {
    "measure": _MEASURE_RULES,
    "radii": _MEASURE_RULES.replace(
        'the nominal value as printed (e.g. "Ø36 H8", "78", "20")',
        'the radius as printed (e.g. "R3", "R42")',
    ),
    "gdt": (
        "This crop is ONE GD&T (Geometric Dimensioning & Tolerancing) feature control frame.\n"
        "GD&T symbol -> characteristic (use the English name):\n"
        "  straightness, flatness(⏥/▱), circularity, cylindricity, profile_line, profile_surface,\n"
        "  perpendicularity(⊥), angularity, parallelism, position(⌖/⊕), concentricity, symmetry,\n"
        "  circular_runout, total_runout\n"
        "Modifiers (circled letter): Ⓜ=M(MMC), Ⓛ=L(LMC), Ⓢ=S(RFS), Ⓟ=P(projected).\n"
        "Return STRICT JSON with keys exactly:\n"
        '  "characteristic" - english name above, or null\n'
        '  "symbol"         - the glyph as printed (e.g. "⌖","⊥","▱")\n'
        '  "tolerance"      - zone value as printed (e.g. "Ø0.4","0.08")\n'
        '  "modifier"       - "M"|"L"|"S"|"P"|null (modifier on the tolerance value)\n'
        '  "datums"         - ordered list WITH modifiers, e.g. ["A","B(M)","C"]; [] if none\n'
        '  "raw"            - verbatim transcription, left to right\n'
        "Rules: copy VERBATIM; preserve European commas; a circled letter on a datum stays inside "
        'that entry (B+Ⓜ -> "B(M)"). Output ONLY the JSON object.'
    ),
    # 표제란 5필드 — extract_title_block.py 의 정의와 동일하게 유지.
    "titleblock": (
        "This image is the title block of a mechanical engineering drawing.\n"
        "Extract exactly these five fields as STRICT JSON with keys: "
        '"Title", "Drawing No.", "LIC. Material", "Material", "Rev".\n'
        '- "Rev": value at the BOTTOM of the narrow far-right "Rev" column (not the "Security" cell).\n'
        "- Copy text verbatim. If a field cannot be found, set it to null.\n"
        "- Output ONLY the JSON object."
    ),
}

# 알려진 재질 코드의 표준 표기(대소문자만 교정) — extract_title_block.py 와 동일.
CANONICAL_MATERIALS = ["SCr18N8", "SUS316L", "SCrNi"]
