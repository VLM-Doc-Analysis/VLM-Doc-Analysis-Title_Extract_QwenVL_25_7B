# 도면 정보 추출 파이프라인 정리 (Qwen2.5-VL + YOLO/OBB)

> 작성일: 2026-06-14
> 대상 프로젝트: `/home/jhkim/projects/Title_Qwen_VL_25_7B`
> 목적: 기계 도면에서 **표제란 5개 필드** + **치수/공차/GD&T/반경**을 정확하게 추출하고, 할루시네이션을 줄이는 방법 정리.

---

## 0. 한눈에 보기

| 작업 | 방법 | 산출물 |
|---|---|---|
| 표제란 5필드 추출 | Qwen2.5-VL 단독 (전체 이미지 → JSON) | `*_title_block.json` |
| 표제란 견고화(선택) | YOLO 단일 `cell` 검출 → 규칙 매핑 → Qwen | `*_title_block_yolo.json` |
| 치수·공차·GD&T·반경 | YOLO **OBB** 검출 → deskew 크롭 → Qwen 구조화 | `*_annotations.json` |

핵심 아이디어 **"검출(어디에) → 크롭 → VLM(무엇이라고)"** 는 세 작업 모두 동일.
역할 분리로 정확도↑·환각↓: **위치는 검출기/코드가, 글자 읽기는 VLM이** 담당.

---

## 1. 베이스라인: 표제란 5필드 추출 (Qwen2.5-VL 단독)

`extract_title_block.ipynb` 섹션 1~5.

- 모델: `Qwen/Qwen2.5-VL-7B-Instruct` (한 번 로드 후 재사용)
- 추출 필드: `Title`, `Drawing No.`, `LIC. Material`, `Material`, `Rev`
- 프롬프트 규칙: 5개 키만 사용 / verbatim 복사 / 못 찾으면 null / **JSON만 출력**
- 후처리:
  - `parse_json()` — 출력에서 첫 번째 `{...}`만 안전하게 파싱(코드펜스 제거)
  - `fix_material_case()` — 알려진 재질 코드의 **대소문자만** 교정 (예: `SCR18N8` → `SCr18N8`)
- 추론: `do_sample=False`(그리디, 결정적) + `torch.no_grad()`

### 이 방식의 한계 (→ 개선 동기)
- 모델이 "어느 칸이 무슨 필드인지"를 **스스로 추론** → `Rev` vs `Security` 혼동 같은 오류
- 작은 글씨(Rev, 재질 코드)는 전체 이미지에서 읽기 어려움
- 양식이 바뀌면 프롬프트의 위치 설명("맨 오른쪽 좁은 칸")이 깨질 수 있음

---

## 2. 표제란 견고화(선택): YOLO 단일 `cell` 클래스 + 규칙 매핑

`extract_title_block.ipynb` 섹션 6.

### 왜 "5개 필드 클래스"가 아니라 "단일 cell 클래스"인가
- 표제란 칸들은 **생김새가 거의 동일** → YOLO가 생김새로 5종을 구분하기 어려움(클래스 혼동).
- 구분되는 단서는 오직 **위치**인데, 위치는 신경망보다 **좌표 비교 코드**가 100% 정확.

### 역할 분리
| 잘하는 주체 | 담당 |
|---|---|
| YOLO (생김새) | "글자 칸이 어디 있나" — 박스 검출 |
| 코드 (위치) | "그 칸이 무슨 필드인가" — 행/열 정렬 + 규칙 매핑 |
| Qwen | "그 칸의 글자" — 크롭 받아쓰기 |

### 매핑 로직 (코드)
1. **행(row) 묶기**: y 중심이 비슷하면 같은 줄 → 위→아래 정렬
2. **열(column) 정렬**: 같은 행 안에서 x 기준 왼→오른
3. **필드 부여**: 위치 규칙 + **항목명 앵커**(예: "LIC." 글자가 있는 칸의 옆/아래가 값)
4. `Rev` 처럼 항목명이 없는 건 위치 규칙(맨 오른쪽 열 최하단)으로 보완

> 장점: 필드 정의가 바뀌어도 **YOLO 재학습 없이 매핑 규칙(코드)만** 수정.
> 단, 양식이 1~2종 고정이면 YOLO조차 불필요 → **고정 ROI 크롭**이 더 간단·정확.

---

## 3. 치수/공차/GD&T/반경 추출: YOLO **OBB** 파이프라인

`extract_title_block.ipynb` 섹션 7. 참고 다이어그램: `doc_sample_paper.jpg`.

### 표제란과 다른 점 3가지
1. **OBB(Oriented Bounding Box, 회전 박스)**
   치수/반경/GD&T는 기울어져 인쇄되는 경우가 많음(예: `R42`). 수평 박스(AABB)로 자르면
   옆 글자가 섞임 → **회전 박스로 검출 후 수평으로 펴서(deskew) 크롭**.
2. **3종 클래스** (도면 범례 `doc_sample_paper.jpg`와 동일)
   - `measure` — 치수값+공차 (`Ø36 H8`, `78`, `18 ±0,1`, `20 +0,3/0`)
   - `gdt` — 기하공차 프레임 (`⊕ Ø0.4Ⓜ A BⓂ C`, `▱ 0.08`, `⊥ 0.1Ⓜ A` + 데이텀 A·B·C)
   - `radii` — 반경 (`R3`, `R42`)
3. **클래스별 구조화 출력** (Qwen이 크롭마다 strict JSON)

### 파이프라인 흐름
```
도면 이미지
   │
   ▼  detect_obb()        ← YOLO-OBB, 박스마다 poly(4코너) + xywhr(중심·크기·회전각)
[OBB 검출 (measure/gdt/radii)]
   │
   ▼  deskew_crop()       ← 회전각 보정 후 수평으로 펴서 크롭 + 확대
[클래스별 회전보정 크롭]
   │
   ▼  read_measure() / read_gdt()   ← 클래스별 프롬프트로 Qwen 구조화 읽기
[strict JSON 파싱]
   │
   ▼
{measures[], radii[], gdt[], counts}
```

### 출력 스키마 예시
```json
{
  "measures": [
    {"nominal": "Ø36 H8", "upper": null, "lower": null, "raw": "Ø36 H8", "bbox": [...], "conf": 0.91},
    {"nominal": "18", "upper": "+0,1", "lower": "-0,1", "raw": "18 ±0,1", "bbox": [...], "conf": 0.88},
    {"nominal": "20", "upper": "+0,3", "lower": "0",    "raw": "20 +0,3 0", "bbox": [...], "conf": 0.87}
  ],
  "radii": [
    {"nominal": "R3",  "upper": null, "lower": null, "raw": "R3",  "bbox": [...], "conf": 0.9},
    {"nominal": "R42", "upper": null, "lower": null, "raw": "R42", "bbox": [...], "conf": 0.86}
  ],
  "gdt": [
    {"characteristic": "position", "symbol": "⌖", "tolerance": "Ø0.4",
     "modifier": "M", "datums": ["A", "B(M)", "C"], "raw": "...", "bbox": [...], "conf": 0.88}
  ],
  "counts": {"measures": 3, "radii": 2, "gdt": 1}
}
```

### 검증/조정 포인트 (코드 내 TODO)
- `deskew_crop`의 **회전 부호**: 글자가 거꾸로/반대로 펴지면 `deg → -deg`
- 세로쓰기 박스 **90° 회전 방향**(시계/반시계)
- `CLASS_NAMES` 인덱스 ↔ `drawing_obb.yaml` names 순서 **일치**
- 소수점 **쉼표 표기**(`±0,1`, `+0,3`)는 `raw`에 그대로 보존

---

## 4. YOLO-OBB 학습 셋업

### 생성된 파일/폴더
```
Title_Qwen_VL_25_7B/
├── drawing_obb.yaml          # 데이터셋 설정 (names: 0=measure 1=gdt 2=radii)
├── train_obb.py              # 학습/검증/추론 스크립트
└── datasets/drawing_obb/
    ├── images/{train,val}/   # 도면 이미지
    └── labels/{train,val}/   # OBB 라벨(.txt)
```

### 설치 (kardi_env)
```bash
# torch 2.11+cu130 을 보존하기 위해 --no-deps 로 설치
/home/jhkim/anaconda3/envs/kardi_env/bin/pip install ultralytics --no-deps
/home/jhkim/anaconda3/envs/kardi_env/bin/pip install opencv-python pyyaml pandas requests matplotlib psutil py-cpuinfo
```

### 라벨 포맷 (OBB)
`labels/<split>/<이름>.txt`, 한 줄 = 박스 하나:
```
class  x1 y1 x2 y2 x3 y3 x4 y4      # 0~1 정규화, class: 0=measure 1=gdt 2=radii
```
- 라벨링 도구: Roboflow / X-AnyLabeling / labelImg(OBB) → "YOLO OBB" 포맷 export
- `images/train/도면1.jpg` ↔ `labels/train/도면1.txt` (파일명 매칭)
- 권장: 클래스당 수백 박스. `radii`/`gdt`는 수가 적어 **불균형 주의**.

### 학습 명령
```bash
cd /home/jhkim/projects/Title_Qwen_VL_25_7B
# 스크립트 방식(도면 특화 증강 포함)
/home/jhkim/anaconda3/envs/kardi_env/bin/python train_obb.py
# 더 큰 모델: --model yolo11s-obb.pt --epochs 150

# CLI 한 줄 (동일)
yolo obb train data=drawing_obb.yaml model=yolo11n-obb.pt \
  imgsz=1280 epochs=100 batch=-1 device=0 \
  project=yolo_obb_drawing name=train \
  degrees=10 fliplr=0 flipud=0 hsv_h=0 hsv_s=0 mosaic=0.5

# 검증 / 추론
python train_obb.py --mode val
python train_obb.py --mode predict --source <도면.jpg>
```

### 주요 하이퍼파라미터 의도
| 설정 | 값 | 이유 |
|---|---|---|
| `model` | `yolo11n-obb.pt` | OBB 사전학습 전이. 데이터 적으면 n/s |
| `imgsz` | `1280` | 도면 글자가 작아 고해상도 필수 (부족 시 960) |
| `fliplr`/`flipud` | `0` | 반전 끔 — `R`·숫자·심볼 **방향이 의미** |
| `degrees` | `10` | OBB라 약한 회전 증강은 도움 |
| `hsv_*` | 최소 | 흑백 선화 → 색 변형 무의미 |
| `batch` | `-1` | GPU 메모리에 맞춰 자동 |

### 학습 후 연결
- 결과: `yolo_obb_drawing/train/weights/best.pt`
- 노트북 **셀 7-1 `OBB_WEIGHTS`** 가 이 경로를 가리킴 → 셀 7-5 실행 시 전체 파이프라인 동작

---

## 5. 주의사항 / 함정

- **`doc_sample_paper.jpg` 는 학습 데이터가 아님**: 박스·범례·3D 모델이 이미 합성된 *설명용 다이어그램*. 학습/추론에 쓰면 박스·범례까지 인식해 버림. → **박스 없는 원본 도면**을 라벨링할 것. (단, 그 안의 패치 값들은 좋은 *정답 예시*로 활용 가능)
- **클래스 인덱스 정합성**: `drawing_obb.yaml`의 `names` 순서 = 노트북 `CLASS_NAMES` 인덱스. 어긋나면 라벨이 뒤바뀜.
- **환경 경로**: macOS 경로(`/Users/...`)는 이 리눅스 머신에서 접근 불가. 파일은 프로젝트 폴더로 옮길 것.
- **GD&T 심볼**: VLM은 심볼 문자(⌖, ⏥, ⟂)보다 영문 명칭(`characteristic`)을 더 안정적으로 맞히는 경향 → 둘 다 받도록 설계.
- **소수점 쉼표**: 유럽식 `0,1` 표기를 `raw`에 보존(임의로 `.`으로 바꾸지 않음).
- **torch 보존**: `pip install ultralytics` 시 torch가 교체될 수 있어 `--no-deps` 사용.

---

## 6. 다음 단계 (확장 아이디어)

- **치수 ↔ 형상 연관(association)**: 저장된 `bbox`를 이용해 치수값을 치수선/형상에 위치 기반으로 연결.
- **고정 ROI 모드**: 표제란 양식이 1~2종 고정이면 학습 없이 좌표 크롭으로 대체(가장 저비용).
- **검증 세트 자동 평가**: `doc_sample_paper.jpg`의 패치 값을 정답으로 두고 파이프라인 정확도 측정.
- **불균형 보정**: `radii`/`gdt`가 많은 도면 추가 수집 또는 클래스 가중/오버샘플링.

---

## 부록: 관련 파일

| 파일 | 설명 |
|---|---|
| `extract_title_block.ipynb` | 전체 노트북 (섹션 1~5 표제란, 6 단일 cell, 7 OBB) |
| `drawing_obb.yaml` | YOLO-OBB 데이터셋 설정 |
| `train_obb.py` | OBB 학습/검증/추론 스크립트 |
| `doc_sample_paper.jpg` | 방법 설명용 참고 다이어그램 (학습 데이터 아님) |
| `datasets/drawing_obb/` | 학습 이미지·라벨 배치 위치 |
