# donut_data — Donut 학습용 데이터 라벨링 파이프라인

도면 요소(measure / radii / gdt / titleblock)를 **Donut 학습 포맷(이미지+JSON)**으로
만들어 주는 파이프라인. 흐름:

```
[PDF/이미지] → (YOLO-OBB 검출·deskew 크롭  |  미리 자른 크롭)
            → Qwen 자동라벨(부트스트랩)  → Donut 포맷 + train/val/test 분할 + 검수 매니페스트
```

> 설계 근거: `resource/도면요소추출_방법비교.md` 0-A — "대형 VLM(Qwen)으로 자동 라벨링 →
> Donut 학습데이터 축적 → fine-tuned Donut이 주석(특히 GD&T)에서 SOTA(F1 0.965)".

## 개념 요약 (왜·어떻게) — 비교문서 §13~15

이 파이프라인이 왜 "Qwen으로 라벨 → Donut 학습" 구조인지, Donut이 JSON을 어떻게 만드는지의 배경.
(전문: [`../resource/도면요소추출_방법비교.md`](../resource/도면요소추출_방법비교.md) §13~15)

- **§13 — 왜 Donut은 파인튜닝이 필요하고 Qwen은 zero-shot으로 되나**
  Donut은 *고정 task 재현형*(텍스트 지시 입력 통로가 없어 `task_prompt`로 본 스키마만 출력) → **새 스키마(5필드·GD&T)는 예시로 학습해야** 함.
  Qwen은 *instruction-tuned*라 프롬프트만 바꾸면 새 스키마 대응. → **그래서 이 파이프라인은 Qwen으로 정답을 만들어 Donut을 가르친다.**
- **§15 — Donut이 JSON을 만드는 방식**
  크롭 1장 → (Swin→BART) → `task_prompt`로 시작하는 **XML식 토큰 시퀀스** 생성 → `token2json`으로 **평면 dict** 복원.
  **페이지 배열 구조(measures[]·gdt[]…)는 Donut이 아니라 코드가 검출 클래스/bbox로 조립.**
  → 본 파이프라인이 만드는 라벨 `{"task":"<s_gdt>", ...필드}`이 바로 그 **학습 타깃**(아래 출력 구조 참조).
- **§14 — ROI 전처리는 양식 의존적**
  표제란 ROI·Rev크롭·빈칸판별은 *양식 가정*에 의존 → 새 양식엔 재조정 필요(주로 표제란 쪽 이슈).
  본 파이프라인은 *주석(measure/gdt/radii) 학습* 중심이라 영향이 작지만, detect 모드 검출기도 양식이 크게 다르면 재학습 필요.

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
3. 검수가 끝나면 아래 `to_donut_vml.py` 로 donut_vml 학습 포맷으로 변환.

## donut_vml 로 학습 연결 (`to_donut_vml.py`)

donut_vml 의 `DonutDataset`(로컬 모드)은 라벨 dict **전체**를 `json2token` 하므로
보조키 `"task"` 를 제거해야 하고, `build_model_and_processor` 는 `task_prompt` 토큰만
등록하므로 **클래스별 학습**이 무수정으로 가장 안전하다. 변환기가 이를 처리한다.

```bash
# (검수 통과분만) GD&T 학습 데이터 준비
python donut_data/to_donut_vml.py --src datasets/donut_anno \
    --dvml /home/jhkim/projects/donut_vml --class gdt --verified-only
```

수행 내용:
- `datasets/donut_anno/{train,val}` 에서 해당 클래스·`verified==1` 샘플만 골라
- 라벨의 `"task"` 키를 제거(순수 스키마) 후
- `donut_vml/data/processed/drawing_<class>/{train,val}/{images,labels}` 로 복사
- **노트북 Step 1 에 붙여넣을 CFG**(task_prompt·로컬경로·권장 image_size/max_length) 출력

이후 `donut_vml/donut_training.ipynb`:
1. **cell 9(로컬 데이터셋 준비)는 건너뛴다** (이미 정리됨).
2. Step 1 CFG 에 출력된 값을 반영(`dataset_name=None`, `task_prompt=<s_gdt>`, local dirs).
3. 그대로 실행 → `checkpoints/` 에 학습. 평가는 노트북 Step 5(leaf-match).

> **per-class vs unified**: 본 변환기는 클래스별(권장)로 분리한다. 한 모델로 9범주를
> 통합 학습(논문 P1 unified)하려면 라벨에 클래스 토큰을 심고 `build_model_and_processor`
> 가 모든 필드/클래스 토큰을 special token 으로 등록하도록 보강이 필요하다.
> (필드 키에 공백·점이 있으면 — 예 `"Drawing No."` — 토큰이 지저분해지므로,
>  필요시 학습용 키를 `Drawing_No` 처럼 바꾸면 더 깔끔하다.)

## 주의
- `datasets/donut_anno/`, `_crops/`, `_pages/`, `patches/` 는 **사내 도면 파생물**이라
  `.gitignore` 로 커밋 제외(상위 `.gitignore` 참조).
- detect 모드 클래스 인덱스는 `drawing_obb.yaml`(0=measure 1=gdt 2=radii)과 일치해야 함.
