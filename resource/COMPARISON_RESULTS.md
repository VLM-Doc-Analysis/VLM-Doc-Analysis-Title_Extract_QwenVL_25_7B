# 표제란 추출 결과 비교: 전처리 vs 베이스라인(환각)

Qwen2.5-VL-7B로 도면 표제란 5필드(Title / Drawing No. / LIC. Material / Material / Rev)를
추출한 결과를 두 버전으로 비교한다.

- **전처리** (`extract_title_block.ipynb`): ROI 크롭·확대 + Rev/LIC 전용 읽기 적용 → `output/{stem}_title_block.json`
- **베이스라인(환각)** (`extract_title_block_hallusination.ipynb`): 원본 이미지 전체를 그대로 모델에 입력 → `output/{stem}_title_block_baseline.json`

> 디코딩은 greedy(`do_sample=False`)라 재실행해도 결과가 동일하다.

---

## test_title_01 — 차이 없음

| 필드 | 전처리 | 베이스라인(환각) | 일치 |
|---|---|---|---|
| Title | FLANGE, CIRCULAR PLAIN | FLANGE, CIRCULAR PLAIN | ✅ |
| Drawing No. | A14-640003-8 | A14-640003-8 | ✅ |
| LIC. Material | SCr18N8 | SCr18N8 | ✅ |
| Material | SUS316L | SUS316L | ✅ |
| Rev | ["0"] | ["0"] | ✅ |

단순 양식이라 두 버전이 동일하다.

## test_title_02 — 1건 차이

| 필드 | 전처리 | 베이스라인(환각) | 일치 |
|---|---|---|---|
| Title | `FLANGE (JIS 1K-1000A)` | `FLANGE (JIS1K-1000A)` | ❌ |
| Drawing No. | A23-369433-5 | A23-369433-5 | ✅ |
| LIC. Material | W-FU-235-JR | W-FU-235-JR | ✅ |
| Material | SS400 | SS400 | ✅ |
| Rev | ["1","0"] | ["1","0"] | ✅ |

베이스라인은 `JIS 1K` 내부 공백을 누락(`JIS1K`). 전처리(확대)는 공백을 보존한다.

## test_title_03 — 2건 차이

| 필드 | 전처리 | 베이스라인(환각) | 일치 |
|---|---|---|---|
| Title | CLAMP | CLAMP | ✅ |
| Drawing No. | BH3-114983-0 | BH3-114983-0 | ✅ |
| LIC. Material | **null** | **SUS316** | ❌ |
| Material | SUS316 | SUS316 | ✅ |
| Rev | **["6","5","4","3","2","1","0"]** | **["0"]** | ❌ |

환각 2건:

1. **LIC. Material**: 실제 빈 칸인데 베이스라인은 옆 `Material`(SUS316)을 복사. 전처리는 잉크비율 빈칸 판별로 `null` 반환.
2. **Rev**: 좌측 세로 리비전표 6~0(7행)인데 베이스라인은 맨 아래 `0` 하나만. 전처리는 Rev 전용 크롭으로 7개 모두 추출.

---

## 종합

| 도면 | 불일치 필드 | 환각 유형 |
|---|---|---|
| test_title_01 | 0 | — |
| test_title_02 | 1 | Title 공백 소실 |
| test_title_03 | 2 | LIC 빈칸 → 옆값 복사, Rev 누락(7→1) |
| **합계** | **3** | |

전처리 파이프라인(ROI 크롭·확대 + Rev/LIC 전용 읽기)이 베이스라인의 환각 3건을 모두 교정했다.

## 재현 방법

```bash
# 두 노트북을 각각 실행해 JSON 생성 (kardi_env 커널)
jupyter nbconvert --to notebook --execute --allow-errors \
  --ExecutePreprocessor.kernel_name=kardi_env extract_title_block.ipynb
jupyter nbconvert --to notebook --execute --allow-errors \
  --ExecutePreprocessor.kernel_name=kardi_env extract_title_block_hallusination.ipynb
```

이후 `extract_title_block_hallusination.ipynb`의 `compare_results()` 셀(섹션 5b)을 실행하면
이 표와 동일한 비교가 자동 출력된다(모델 불필요, JSON만 읽음).
