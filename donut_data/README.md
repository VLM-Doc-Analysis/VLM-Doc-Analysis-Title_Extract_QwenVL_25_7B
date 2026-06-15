# donut_data — Donut 학습용 데이터 라벨링 파이프라인

도면 요소(measure / radii / gdt / titleblock)를 **Donut 학습 포맷(이미지+JSON)**으로
만들어 주는 파이프라인. 흐름:

```
[PDF/이미지] → (YOLO-OBB 검출·deskew 크롭  |  미리 자른 크롭)
            → Qwen 자동라벨(부트스트랩)  → Donut 포맷 + train/val/test 분할 + 검수 매니페스트
```

> 설계 근거: `resource/도면요소추출_방법비교.md` 0-A — "대형 VLM(Qwen)으로 자동 라벨링 →
> Donut 학습데이터 축적 → fine-tuned Donut이 주석(특히 GD&T)에서 SOTA(F1 0.965)".

## 파일
| 파일 | 역할 |
|---|---|
| `schemas.py` | 클래스별 정답 JSON 스키마 · `task_prompt` · Qwen 자동라벨 프롬프트 |
| `autolabel.py` | Qwen2.5-VL 라벨러(모델 1회 로드 후 재사용, greedy=결정적) |
| `build_dataset.py` | 오케스트레이터(수집 → 라벨 → Donut 포맷 + 분할 + manifest) |

## 사전 준비 (kardi_env)
```bash
# 자동라벨(Qwen): 이미 설치됨(transformers, qwen-vl-utils)
# detect 모드만 추가 필요:
/home/jhkim/anaconda3/envs/kardi_env/bin/pip install ultralytics --no-deps
# cv2, pdf2image 는 설치되어 있음
```

## 사용법

**1) 콜드스타트 (YOLO 없이, 권장 시작점)** — 손으로 몇 장 잘라 클래스별 폴더에 두고 자동라벨:
```
patches/
  measure/*.png   radii/*.png   gdt/*.png   titleblock/*.png
```
```bash
python donut_data/build_dataset.py --mode patches --input patches \
    --out datasets/donut_anno --classes measure radii gdt titleblock
```

**2) YOLO 학습 후 전체 도면에서 자동 검출·크롭·라벨:**
```bash
python donut_data/build_dataset.py --mode detect --input input_doc \
    --weights yolo_obb_drawing/train/weights/best.pt --out datasets/donut_anno
```

**3) 모델 없이 빈 템플릿만(순수 수작업 라벨링용):**
```bash
python donut_data/build_dataset.py --mode patches --input patches --out datasets/donut_anno --no-model
```

**few-shot 정확도 보강(선택):** `shots/<class>/<name>.png` + `shots/<class>/<name>.json`(정답) 두고
`--shots shots` 추가하면 in-context 예시로 자동라벨 품질↑.

## 출력 구조 (donut_vml 의 DonutDataset(local) 과 호환)
```
datasets/donut_anno/
  manifest.csv                 # 검수용: id,class,split,autolabeled,parse_ok,verified,raw
  train/images/*.png  train/labels/*.json
  val/images/*.png    val/labels/*.json
  test/images/*.png   test/labels/*.json
```
- 라벨 JSON 예: `{"task":"<s_gdt>","characteristic":"position","tolerance":"Ø0.4","modifier":"M","datums":["A","B(M)","C"],"raw":"..."}`

## 검수 → 학습
1. `manifest.csv` 에서 `parse_ok=0`(파싱 실패=빈 템플릿) 행부터 확인.
2. 각 크롭을 보고 `labels/<id>.json` 수정 후, `verified` 를 1 로.
3. 검수가 끝나면 `donut_vml` 의 Donut 파인튜닝(로컬 데이터셋 모드)에 이 루트를 지정해 학습.
   - 학습 노트북의 `task_prompt` 는 클래스별로 분리하거나, 라벨의 `"task"` 값을 사용.

## 주의
- `datasets/donut_anno/`, `_crops/`, `_pages/`, `patches/` 는 **사내 도면 파생물**이라
  `.gitignore` 로 커밋 제외(상위 `.gitignore` 참조).
- detect 모드 클래스 인덱스는 `drawing_obb.yaml`(0=measure 1=gdt 2=radii)과 일치해야 함.
