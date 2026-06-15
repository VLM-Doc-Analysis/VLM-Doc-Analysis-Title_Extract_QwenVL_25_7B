"""Qwen2.5-VL 기반 크롭 자동 라벨러 (Donut 학습데이터 부트스트랩).

콜드스타트(실데이터 소량)에서 사람 라벨링 부담을 줄이려, 잘라낸 요소 크롭을
Qwen zero/few-shot 으로 1차 라벨링(JSON)한 뒤 → 사람이 검수하는 흐름을 지원한다.
(비교문서 0-A: "대형 VLM으로 자동 라벨링 -> Donut 학습데이터 축적")

모델은 한 번만 로드해 여러 크롭에 재사용한다. greedy 디코딩이라 결정적.
"""
import json
import os
import re

import schemas


def parse_json(text: str):
    """모델 출력에서 첫 {...} JSON 객체만 안전 추출(코드펜스 제거). 실패 시 예외."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    return json.loads(cleaned)


def fix_material_case(value):
    """알려진 재질코드의 대소문자만 표준 표기로 교정(문자 자체는 불변)."""
    if not isinstance(value, str):
        return value
    key = value.strip().casefold()
    for canon in schemas.CANONICAL_MATERIALS:
        if canon.casefold() == key:
            return canon
    return value


class Labeler:
    """Qwen2.5-VL 라벨러. shots(few-shot 예시)는 클래스별 (이미지경로, 정답dict) 리스트."""

    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", shots=None):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.shots = shots or {}
        print(f"[autolabel] loading {model_id} ...", flush=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_id)

    @staticmethod
    def _img(path):
        return {"type": "image", "image": f"file://{os.path.abspath(path)}"}

    def _messages(self, image_path, cls):
        prompt = schemas.PROMPTS[cls]
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for shot_img, shot_gold in self.shots.get(cls, []):       # few-shot(있으면)
            msgs.append({"role": "user",
                         "content": [self._img(shot_img), {"type": "text", "text": "Extract:"}]})
            msgs.append({"role": "assistant",
                         "content": [{"type": "text",
                                      "text": json.dumps(shot_gold, ensure_ascii=False)}]})
        msgs.append({"role": "user",
                     "content": [self._img(image_path), {"type": "text", "text": "Extract:"}]})
        return msgs

    def label(self, image_path, cls):
        """크롭 1장 -> (라벨dict, 원문, 성공여부). 파싱 실패 시 빈 스키마 + ok=False."""
        from qwen_vl_utils import process_vision_info

        messages = self._messages(image_path, cls)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        raw = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        try:
            result = parse_json(raw)
        except json.JSONDecodeError:
            return dict(schemas.EMPTY_SCHEMA[cls]), raw, False

        # 스키마 키 정렬 + 표제란 재질 대소문자 교정
        out = dict(schemas.EMPTY_SCHEMA[cls])
        for k in out:
            if k in result:
                out[k] = result[k]
        if cls == "titleblock":
            for f in ("LIC. Material", "Material"):
                out[f] = fix_material_case(out[f])
        return out, raw, True
