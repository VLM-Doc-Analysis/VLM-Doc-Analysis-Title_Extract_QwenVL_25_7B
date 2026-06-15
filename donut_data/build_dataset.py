"""Donut 학습용 데이터 라벨링 파이프라인 (orchestrator).

흐름:  [PDF/이미지] → (YOLO-OBB 검출·deskew 크롭 | 또는 미리 자른 크롭) →
       Qwen 자동라벨(부트스트랩) → Donut 포맷(images/+labels/) + train/val/test 분할 + 검수 매니페스트

두 입력 모드:
  --mode patches : 미리 잘라둔 크롭을 클래스별 폴더로 둔 경우(YOLO 불필요, 콜드스타트 권장)
        <input>/<class>/*.png        (class ∈ measure radii gdt titleblock)
  --mode detect  : 전체 도면(PDF/PNG) + 학습된 YOLO-OBB 가중치로 자동 검출·크롭
        (ultralytics 필요. drawing_obb.yaml 의 0=measure 1=gdt 2=radii 와 인덱스 일치)

사용 예:
  # 1) 콜드스타트: 손으로 몇 장 잘라 patches/<class>/ 에 두고 자동라벨
  python donut_data/build_dataset.py --mode patches --input patches \
      --out datasets/donut_anno --classes measure radii gdt

  # 2) YOLO 학습 후 전체 도면에서 자동 검출·크롭·라벨
  python donut_data/build_dataset.py --mode detect --input input_doc \
      --weights yolo_obb_drawing/train/weights/best.pt --out datasets/donut_anno

  # 3) 모델 없이 빈 템플릿만 생성(순수 수작업 라벨링용)
  python donut_data/build_dataset.py --mode patches --input patches --out datasets/donut_anno --no-model

검수: out/manifest.csv 의 verified 열(0)을 사람이 확인 후 1 로 바꾸고, 필요시 labels/*.json 수정.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # schemas/autolabel 직접 import
import schemas

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ── 입력 수집 ────────────────────────────────────────────────────────────────
def collect_patches(input_dir, classes):
    """patches 모드: <input>/<class>/*.{img} 를 (이미지경로, class) 리스트로."""
    items = []
    for cls in classes:
        d = os.path.join(input_dir, cls)
        if not os.path.isdir(d):
            print(f"  [skip] 폴더 없음: {d}")
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(IMG_EXTS):
                items.append((os.path.join(d, fn), cls))
    return items


def pdfs_to_images(input_dir, work_dir, dpi=300):
    """input_dir 의 PDF 를 work_dir 에 PNG 로 변환하고, PNG 경로 + 원래 이미지들을 합쳐 반환."""
    from pdf2image import convert_from_path

    os.makedirs(work_dir, exist_ok=True)
    pages = []
    for fn in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, fn)
        stem, ext = os.path.splitext(fn)
        if ext.lower() == ".pdf":
            for i, img in enumerate(convert_from_path(path, dpi=dpi)):
                out = os.path.join(work_dir, f"{stem}_p{i}.png")
                img.save(out)
                pages.append(out)
        elif ext.lower() in IMG_EXTS:
            pages.append(path)
    return pages


def detect_and_crop(image_paths, weights, out_crop_dir, classes, conf=0.25, pad=0.06):
    """detect 모드: YOLO-OBB 로 검출 → minAreaRect deskew 크롭 → out_crop_dir/<class>/ 저장."""
    import cv2
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(weights)
    names = model.names                       # {0:'measure',1:'gdt',2:'radii'}
    items = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [warn] 읽기 실패: {img_path}");  continue
        res = model.predict(img_path, conf=conf, verbose=False)[0]
        if res.obb is None or len(res.obb) == 0:
            continue
        polys = res.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)   # (N,4,2)
        clsids = res.obb.cls.cpu().numpy().astype(int)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        for k, (poly, cid) in enumerate(zip(polys, clsids)):
            cls = names.get(int(cid), str(cid))
            if cls not in classes:
                continue
            crop = _deskew_crop(img, poly, cv2, np, pad)
            if crop is None or crop.size == 0:
                continue
            d = os.path.join(out_crop_dir, cls);  os.makedirs(d, exist_ok=True)
            out = os.path.join(d, f"{stem}_{k:03d}.png")
            cv2.imwrite(out, crop)
            items.append((out, cls))
    return items


def _deskew_crop(img, poly, cv2, np, pad):
    """OBB 4코너 → 회전 보정해 수평으로 펴서 크롭(약간 패딩)."""
    rect = cv2.minAreaRect(poly.astype(np.float32))     # ((cx,cy),(w,h),angle)
    (cx, cy), (w, h), angle = rect
    if w < 2 or h < 2:
        return None
    w2, h2 = int(w * (1 + pad)), int(h * (1 + pad))
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    crop = cv2.getRectSubPix(rotated, (w2, h2), (cx, cy))
    if crop is not None and crop.shape[0] > crop.shape[1]:   # 세로형이면 가로로 회전
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


# ── 분할 ─────────────────────────────────────────────────────────────────────
def split_index(n_items, ratios, seed=42):
    """결정적 분할: 정렬된 인덱스를 비율대로 train/val/test 에 배정."""
    import random
    idx = list(range(n_items))
    random.Random(seed).shuffle(idx)
    n_tr = int(n_items * ratios[0])
    n_va = int(n_items * ratios[1])
    assign = {}
    for j, i in enumerate(idx):
        assign[i] = "train" if j < n_tr else ("val" if j < n_tr + n_va else "test")
    return assign


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Donut 학습용 라벨링 파이프라인")
    ap.add_argument("--mode", choices=["patches", "detect"], default="patches")
    ap.add_argument("--input", required=True, help="patches: 크롭 루트 / detect: 도면(PDF·PNG) 폴더")
    ap.add_argument("--out", required=True, help="Donut 데이터셋 출력 루트")
    ap.add_argument("--classes", nargs="+", default=["measure", "radii", "gdt"],
                    choices=schemas.CLASSES)
    ap.add_argument("--weights", default="yolo_obb_drawing/train/weights/best.pt",
                    help="detect 모드 YOLO-OBB 가중치")
    ap.add_argument("--dpi", type=int, default=300, help="detect 모드 PDF 래스터화 해상도")
    ap.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--shots", default=None,
                    help="few-shot 예시 루트(shots/<class>/*.png + 같은이름.json). 선택")
    ap.add_argument("--no-model", action="store_true", help="자동라벨 생략, 빈 템플릿만 생성")
    args = ap.parse_args()

    # 1) 입력 크롭 수집
    if args.mode == "patches":
        items = collect_patches(args.input, args.classes)
    else:
        work = os.path.join(args.out, "_pages")
        pages = pdfs_to_images(args.input, work, dpi=args.dpi)
        print(f"[detect] 페이지 {len(pages)}장 → YOLO-OBB 검출/크롭 ...")
        crop_dir = os.path.join(args.out, "_crops")
        items = detect_and_crop(pages, args.weights, crop_dir, args.classes)
    if not items:
        sys.exit("크롭이 없습니다. 입력 경로/클래스 폴더를 확인하세요.")
    print(f"[수집] 크롭 {len(items)}개")

    # 2) (선택) few-shot 예시 로드
    shots = _load_shots(args.shots, args.classes) if args.shots else None

    # 3) 라벨러 준비
    labeler = None
    if not args.no_model:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import autolabel
        labeler = autolabel.Labeler(model_id=args.model_id, shots=shots)

    # 4) 분할 배정 + 디렉토리
    assign = split_index(len(items), args.split)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(args.out, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.out, sp, "labels"), exist_ok=True)

    # 5) 라벨링 + 저장 + 매니페스트
    import shutil
    manifest_path = os.path.join(args.out, "manifest.csv")
    rows, n_ok, n_fail = [], 0, 0
    for i, (img_path, cls) in enumerate(items):
        sp = assign[i]
        uid = f"{cls}_{i:05d}"
        ext = os.path.splitext(img_path)[1].lower()
        dst_img = os.path.join(args.out, sp, "images", uid + ext)
        dst_lbl = os.path.join(args.out, sp, "labels", uid + ".json")
        shutil.copy(img_path, dst_img)

        if labeler is None:
            label, raw, ok = dict(schemas.EMPTY_SCHEMA[cls]), "", False
        else:
            label, raw, ok = labeler.label(img_path, cls)
        n_ok += int(ok);  n_fail += int(not ok)

        # Donut 라벨: task_prompt + 필드들을 한 객체로(학습 노트북의 json2token 이 토큰화)
        out_obj = {"task": schemas.TASK_PROMPT[cls], **label}
        with open(dst_lbl, "w", encoding="utf-8") as fh:
            json.dump(out_obj, fh, ensure_ascii=False, indent=2)

        rows.append({
            "id": uid, "class": cls, "split": sp,
            "src": os.path.relpath(img_path, args.out) if img_path.startswith(args.out) else img_path,
            "autolabeled": int(labeler is not None),
            "parse_ok": int(ok),
            "verified": 0,                          # ← 사람이 검수 후 1 로
            "raw": (raw[:120].replace("\n", " ") if raw else ""),
        })
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(items)}")

    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader();  w.writerows(rows)

    # 6) 요약
    n = len(items)
    by_split = {sp: sum(1 for r in rows if r["split"] == sp) for sp in ("train", "val", "test")}
    print("\n[완료]")
    print(f"  데이터셋 루트 : {args.out}")
    print(f"  크롭 총 {n}개  train/val/test = {by_split['train']}/{by_split['val']}/{by_split['test']}")
    if labeler is not None:
        print(f"  자동라벨 파싱 성공 {n_ok} / 실패 {n_fail}  (실패분은 빈 템플릿 → 수작업)")
    print(f"  검수 매니페스트: {manifest_path}  (verified=1 로 표시하며 검토)")
    print("  학습: donut_vml 의 DonutDataset(local) 로 이 루트를 가리키면 됨"
          " (images/ + labels/<stem>.json).")


def _load_shots(shots_root, classes):
    shots = {}
    for cls in classes:
        d = os.path.join(shots_root, cls)
        if not os.path.isdir(d):
            continue
        pairs = []
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(IMG_EXTS):
                jp = os.path.join(d, os.path.splitext(fn)[0] + ".json")
                if os.path.isfile(jp):
                    with open(jp, encoding="utf-8") as fh:
                        pairs.append((os.path.join(d, fn), json.load(fh)))
        if pairs:
            shots[cls] = pairs
    return shots


if __name__ == "__main__":
    main()
