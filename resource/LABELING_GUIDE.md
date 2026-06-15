# 라벨링 가이드 — 도면 검출기 학습 데이터

이 프로젝트는 검출기가 **2종**이다. 각각 라벨 포맷·대상이 다르다.

| 검출기 | 스크립트 / 설정 | 클래스 | 라벨 포맷 |
|---|---|---|---|
| **레이아웃 det** (영역) | `train_layout.py` / `drawing_layout.yaml` | 0=title_block 1=view 2=note | **HBB**: `class cx cy w h` |
| **주석 OBB** (회전) | `train_obb.py` / `drawing_obb.yaml` | 0=measure 1=gdt 2=radii | **OBB**: `class x1 y1 x2 y2 x3 y3 x4 y4` |

좌표는 모두 **이미지 크기로 0~1 정규화**. `images/<split>/X.png` ↔ `labels/<split>/X.txt` **파일명 stem 일치**.

> 📎 **복사용 .txt 예시**: [`label_samples/`](label_samples/) — `layout_example.txt`(det),
> `obb_example.txt`(OBB), 각 줄 설명은 `label_samples/README.md`.

---

## 0. 도구

| 도구 | OBB 지원 | 비고 |
|---|---|---|
| **X-AnyLabeling** | ✅ | 로컬, HBB+OBB 모두. "YOLO/YOLO-OBB" export 권장 |
| **Roboflow** | ✅ | 웹, 쉬움, YOLO/YOLO-OBB export |
| CVAT | ✅ | 로컬/서버, 대규모 |
| labelImg | ✕(HBB만) | 레이아웃 det에만. OBB는 roLabelImg/X-AnyLabeling |

- export 시 **클래스 순서 = yaml `names` 순서**와 반드시 일치(어긋나면 라벨 뒤바뀜).
- export → `datasets/<drawing_layout|drawing_obb>/{images,labels}/{train,val}/` 로 배치.

---

## 1. 레이아웃 det 라벨링 (title_block / view / note)

**포맷(HBB)**: 한 줄 = `class cx cy w h` (중심좌표·폭·높이, 0~1). 예:
```
0 0.78 0.90 0.40 0.18      # title_block (우하단)
1 0.30 0.40 0.45 0.55      # view (정면도)
2 0.62 0.80 0.20 0.06      # note
```

### class 0 — title_block (표제란)
- **무엇**: Title / Drawing No. / 재질 / Rev 등이 담긴 **우하단 정형 표 블록 전체**.
- **경계 규칙**: 표제란 표의 바깥 테두리까지 포함. **Rev 열·리비전 매트릭스가 표제란에 붙어 있으면 함께 포함**(Rev 값이 거기 있음). 부품표가 별도 큰 표로 떨어져 있으면 제외.
- 도면당 보통 **1개**. (세로형=하단 띠, 가로형=우하단 — 위치는 다양해도 "그 표 전체"를 박싱)

### class 1 — view (도면 뷰)
- **무엇**: 정면도/단면도(A-A)/등각도/Detail 등 **각 뷰 1개씩**.
- **경계 규칙**: 형상 + **그 뷰에 속한 치수·기호까지 포함**되게 약간 넉넉히(이 박스 안에서 주석 OBB를 검출하므로). 인접 뷰와 겹치면 겹쳐도 됨(IoU 무관, 각 뷰 독립 박스).
- 도면당 **여러 개**(P2 평균 도면당 뷰 다수).

### class 2 — note (노트/주기)
- **무엇**: 자유문 주기 블록(예: `NOTE: 1. Sharp edges to be removed ...`).
- **경계 규칙**: 텍스트 단락 전체를 한 박스로. 여러 노트 블록이면 각각.

> 팁: title_block은 항상 라벨(가장 중요). view/note는 주석·노트 추출까지 갈 때 라벨.
> **표제란만 필요**하면 0번만 라벨해도 됨(yaml은 그대로, 1·2는 안 써도 무방).

---

## 2. 주석 OBB 라벨링 (measure / gdt / radii)

**포맷(OBB)**: 한 줄 = `class x1 y1 x2 y2 x3 y3 x4 y4` (4코너, 0~1, **시계방향 권장**). 글자가 기울면 박스도 글자 방향으로 회전.

### class 0 — measure (치수 + 공차)
- **무엇**: 치수값(+공차). 예: `Ø36 H8`, `78`, `18 ±0,1`, `20 +0,3/0`.
- **모따기(`1×45°`, `C2` 등)도 measure에 포함**(별도 클래스 없음).
- **경계**: 치수 텍스트(+공차)만 타이트하게. 치수선/화살표는 제외(글자 위주). 기울어진 치수는 **회전 박스**로.

### class 1 — gdt (기하공차 프레임)
- **무엇**: feature control frame **전체**. 예: `⊕ Ø0.4Ⓜ A B C`, `⊥ 0.1Ⓜ A`, `▱ 0.08`.
- **경계**: 기호+공차+수정자+**데이텀 문자까지 한 박스**. 데이텀 정의 심볼(◷A 등)은 별개 취급(필요 시 measure/제외 — 일관되게).

### class 2 — radii (반경)
- **무엇**: 반경 치수. 예: `R3`, `R42`, `R5.0`.
- measure와 구분(접두 `R`). 기울어지면 회전 박스.

> **유럽식 쉼표**(`±0,1`)는 *라벨 텍스트가 아니라* 박스만 그림(텍스트는 VLM이 읽음). OBB는 위치만.
> **클래스 불균형**: radii·gdt는 measure보다 훨씬 적음 → 일부러 더 모으거나 학습 시 가중/오버샘플.

---

## 3. 공통 규칙 (꼭 지킬 것)

- **정규화**: 모든 좌표 0~1 (픽셀값 금지).
- **파일명 매칭**: `images/train/도면1.png` ↔ `labels/train/도면1.txt`.
- **클래스 인덱스 = yaml `names` 순서**. 레이아웃 0/1/2 = title_block/view/note, OBB 0/1/2 = measure/gdt/radii.
- **train/val 분할**: 보통 8:2. **같은 도면이 train·val에 겹치지 않게**(누설 방지).
- 빈 라벨 파일(객체 없는 이미지)은 빈 `.txt`로 둠(배경 학습).

---

## 4. 데이터량·품질

- **레이아웃 det**: 도면 수십~수백 장(title_block은 도면당 1개라 도면 수 = 표제란 수). P2는 1,000 도면.
- **주석 OBB**: P1은 1,367 도면/11,469 패치. 현실적 시작은 수백 박스/클래스, radii·gdt 보강.
- **품질 체크리스트**
  - [ ] 클래스 순서가 yaml과 일치
  - [ ] 좌표 0~1 정규화
  - [ ] images↔labels stem 일치, train/val 도면 비겹침
  - [ ] title_block 박스가 5필드(특히 Rev)를 모두 포함
  - [ ] OBB 회전 방향이 글자와 일치(거꾸로 X)
  - [ ] 모따기=measure, 반경=radii로 분류 일관

## 5. 흔한 실수
- 클래스 순서 뒤바뀜 → 라벨 전부 어긋남(가장 흔함).
- title_block에서 Rev 열을 빠뜨림 → Rev 학습 불가.
- view 박스를 너무 타이트하게 → 그 뷰의 치수가 박스 밖 → 주석 검출 누락.
- OBB를 수평 박스로만 → 기울어진 치수에서 옆 글자 혼입.
- 픽셀 좌표로 저장(정규화 누락).

## 6. 라벨 → 학습 연결
```bash
# 레이아웃 det
python train_layout.py            # datasets/drawing_layout 사용
# 주석 OBB
python train_obb.py               # datasets/drawing_obb 사용
```
학습된 표제란 검출기로 크롭:
```bash
python donut_data/collect_titleblock.py --input <도면> \
    --weights yolo_layout_drawing/train/weights/best.pt --out patches/titleblock --preview
```

> 관련: 검출 전략 배경은 `도면요소추출_방법비교.md` §6·§14, 라벨→Donut 학습은 `donut_data/README.md`.
