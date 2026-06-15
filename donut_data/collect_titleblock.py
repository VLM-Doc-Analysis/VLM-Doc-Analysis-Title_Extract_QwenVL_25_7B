"""도면(PDF/이미지)에서 표제란 크롭을 자동 수집 → patches/titleblock/.

표제란은 보통 페이지 **하단(세로형은 하단 띠, 가로형은 우하단)** 에 있다는 휴리스틱으로
영역을 잘라낸다. (§14: ROI는 양식 의존적 — 같은 HHI 양식엔 잘 맞지만, 양식이 크게 다르면
비율 조정 또는 표제란 검출기 학습 필요. 충분히 모이면 YOLO-det로 자동화 권장.)

사용 예:
    # 도면만 모인 폴더(또는 파일들)를 입력. ※ 학술 PDF 등 비도면을 섞지 말 것
    python donut_data/collect_titleblock.py --input drawings/ --out patches/titleblock
    python donut_data/collect_titleblock.py --input resource/A1464068392.pdf resource/BH311726385.pdf \
        --out patches/titleblock --preview        # preview=크롭 박스 표시 QA 이미지도 저장

    # 양식이 다르면 비율 조정
    python donut_data/collect_titleblock.py --input d/ --portrait-top 0.80 --land-left 0.55 --land-top 0.70

이후: build_dataset.py 로 자동라벨 → 검수 → to_donut_vml.py.
"""
import argparse
import os
import sys

from PIL import Image, ImageDraw

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def iter_inputs(inputs):
    """파일/폴더 목록 → 처리할 (경로, 종류) 리스트. PDF·이미지만."""
    items = []
    for p in inputs:
        if os.path.isdir(p):
            for fn in sorted(os.listdir(p)):
                full = os.path.join(p, fn)
                if fn.lower().endswith(IMG_EXTS) or fn.lower().endswith(".pdf"):
                    items.append(full)
        elif os.path.isfile(p):
            items.append(p)
        else:
            print(f"  [skip] 없음: {p}")
    return items


def render_pages(path, dpi):
    """PDF→PNG 페이지 이미지 리스트, 이미지면 그대로 1장. (stem, PIL.Image) 반환."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if path.lower().endswith(".pdf"):
        from pdf2image import convert_from_path
        pages = convert_from_path(path, dpi=dpi)
        return [(f"{stem}_p{i}" if len(pages) > 1 else stem, im.convert("RGB"))
                for i, im in enumerate(pages)]
    return [(stem, Image.open(path).convert("RGB"))]


def titleblock_box(W, H, portrait_top, land_left, land_top):
    """방향별 표제란 영역 박스(left,top,right,bottom) 추정 — 휴리스틱(검출기 없을 때)."""
    if H >= W:                       # 세로형: 하단 띠(전폭)
        return (0, int(H * portrait_top), W, H)
    return (int(W * land_left), int(H * land_top), W, H)   # 가로형: 우하단


def detect_boxes(detector, pil_img, class_name, conf):
    """학습된 YOLO-det로 title_block 박스들을 (left,top,right,bottom) 리스트로 검출."""
    import numpy as np
    res = detector.predict(np.array(pil_img)[:, :, ::-1], conf=conf, verbose=False)[0]
    names = res.names
    boxes = []
    if res.boxes is not None:
        for b in res.boxes:
            if names.get(int(b.cls[0]), "") == class_name:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def main():
    ap = argparse.ArgumentParser(description="도면에서 표제란 크롭 자동 수집")
    ap.add_argument("--input", nargs="+", required=True, help="도면 파일/폴더(PDF·이미지)")
    ap.add_argument("--out", default="patches/titleblock", help="크롭 저장 폴더")
    ap.add_argument("--dpi", type=int, default=300, help="PDF 래스터화 해상도")
    ap.add_argument("--portrait-top", type=float, default=0.82, help="세로형 하단 띠 시작 y비율")
    ap.add_argument("--land-left", type=float, default=0.58, help="가로형 우하단 시작 x비율")
    ap.add_argument("--land-top", type=float, default=0.74, help="가로형 우하단 시작 y비율")
    ap.add_argument("--preview", action="store_true", help="크롭 박스 표시 QA 이미지도 저장(_preview)")
    # ── 검출기 모드(선택): 학습된 YOLO-det 가중치로 표제란 검출 → 휴리스틱 대체(§14 견고) ──
    ap.add_argument("--weights", default=None,
                    help="title_block 검출기(train_layout.py best.pt). 주면 휴리스틱 대신 검출 사용")
    ap.add_argument("--class-name", default="title_block", help="검출기에서 자를 클래스명")
    ap.add_argument("--conf", type=float, default=0.25, help="검출기 신뢰도 임계값")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    preview_dir = os.path.join(args.out, "_preview")
    if args.preview:
        os.makedirs(preview_dir, exist_ok=True)

    detector = None
    if args.weights:
        from ultralytics import YOLO
        print(f"[검출기] {args.weights} 로드 — 휴리스틱 대신 검출 기반 크롭")
        detector = YOLO(args.weights)

    n = 0
    for path in iter_inputs(args.input):
        try:
            pages = render_pages(path, args.dpi)
        except Exception as e:
            print(f"  [warn] 처리 실패 {path}: {e}")
            continue
        for stem, im in pages:
            W, H = im.size
            if detector is not None:                     # 검출기 모드: 박스 0..N개
                boxes = detect_boxes(detector, im, args.class_name, args.conf)
                if not boxes:
                    print(f"  [miss] {os.path.basename(path)} {stem}: title_block 미검출")
            else:                                         # 휴리스틱 모드: 박스 1개
                boxes = [titleblock_box(W, H, args.portrait_top, args.land_left, args.land_top)]
            for j, box in enumerate(boxes):
                crop = im.crop(box)
                suffix = f"_tb{j}" if len(boxes) > 1 else "_tb"
                out_path = os.path.join(args.out, f"{stem}{suffix}.png")
                crop.save(out_path)
                n += 1
            mode = "검출" if detector is not None else ("세로" if H >= W else "가로")
            print(f"  [{mode}] {os.path.basename(path)} {im.size} → {len(boxes)}개 크롭")
            if args.preview and boxes:
                pv = im.copy()
                d = ImageDraw.Draw(pv)
                for box in boxes:
                    d.rectangle(box, outline=(255, 0, 0), width=max(3, W // 300))
                pv.save(os.path.join(preview_dir, f"{stem}_preview.png"))

    print(f"\n[완료] 표제란 크롭 {n}개 → {args.out}")
    if args.preview:
        print(f"  QA 미리보기: {preview_dir} (크롭 박스가 표제란을 잘 감싸는지 확인 후 비율 조정)")
    print("  다음: python donut_data/build_dataset.py --mode patches --input patches "
          "--out datasets/donut_anno --classes titleblock")
    if n == 0:
        sys.exit("수집된 크롭 0개 — 입력 경로 확인.")


if __name__ == "__main__":
    main()
