"""donut_data 산출물(datasets/donut_anno)을 donut_vml 학습 포맷으로 물려주는 변환기.

donut_vml 의 DonutDataset(local) 은 <root>/{images,labels} 를 읽어 라벨 dict 전체를
json2token 으로 변환한다. 따라서 우리 라벨의 보조키 "task" 는 제거해야 하며,
build_model_and_processor 가 task_prompt 토큰만 등록하므로 **클래스별 학습이 무수정으로 안전**하다.

이 스크립트는:
  1) datasets/donut_anno/{train,val}/labels/*.json 을 읽어
  2) (선택) manifest.csv 의 verified==1 만 필터
  3) 클래스별로 "task" 키를 제거한 라벨 + 이미지를
     donut_vml/data/processed/drawing_<class>/{train,val}/{images,labels} 로 복사
  4) 노트북 Step1 에 붙여넣을 CFG dict 를 출력

사용 예:
  # GD&T 만 학습 준비 (검수 통과분만)
  python donut_data/to_donut_vml.py --src datasets/donut_anno \
      --dvml /home/jhkim/projects/donut_vml --class gdt --verified-only

  # 표제란 5필드
  python donut_data/to_donut_vml.py --src datasets/donut_anno \
      --dvml /home/jhkim/projects/donut_vml --class titleblock
"""
import argparse
import csv
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schemas

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# 클래스별 권장 학습 설정(패치 특성 반영). 노트북 CFG 에 반영하기 위한 힌트.
#   - 주석 크롭(measure/radii/gdt)은 짧고 가로로 김 → max_length 작게, 해상도 중간.
#   - 표제란은 더 큰 영역 → 해상도 크게.
RECO = {
    "measure":    {"image_size": [320, 768], "max_length": 96},
    "radii":      {"image_size": [320, 768], "max_length": 64},
    "gdt":        {"image_size": [320, 960], "max_length": 128},
    "titleblock": {"image_size": [960, 768], "max_length": 160},
}


def load_verified(src):
    """manifest.csv → verified==1 인 id 집합. 없으면 None(=전체 허용)."""
    mp = os.path.join(src, "manifest.csv")
    if not os.path.isfile(mp):
        return None
    ok = set()
    with open(mp, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("verified", "0")).strip() == "1":
                ok.add(row["id"])
    return ok


def find_image(img_dir, stem):
    for ext in IMG_EXTS:
        p = os.path.join(img_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def convert_split(src, split, cls, dst_root, verified):
    """src/<split> → dst_root/<split> 으로 해당 클래스·검수통과 샘플 복사. 복사 수 반환."""
    in_lbl = os.path.join(src, split, "labels")
    in_img = os.path.join(src, split, "images")
    if not os.path.isdir(in_lbl):
        return 0
    out_img = os.path.join(dst_root, split, "images")
    out_lbl = os.path.join(dst_root, split, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    n = 0
    for fn in sorted(os.listdir(in_lbl)):
        if not fn.endswith(".json"):
            continue
        stem = fn[:-5]
        if not stem.startswith(cls + "_"):          # id 규칙: <class>_NNNNN
            continue
        if verified is not None and stem not in verified:
            continue
        with open(os.path.join(in_lbl, fn), encoding="utf-8") as f:
            label = json.load(f)
        label.pop("task", None)                     # 보조키 제거 → 순수 스키마만
        img = find_image(in_img, stem)
        if img is None:
            print(f"  [warn] 이미지 없음: {stem}");  continue
        shutil.copy2(img, os.path.join(out_img, os.path.basename(img)))
        with open(os.path.join(out_lbl, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(label, f, ensure_ascii=False, indent=2)
        n += 1
    return n


def print_cfg(cls, dst_root):
    reco = RECO.get(cls, {"image_size": [1280, 960], "max_length": 256})
    task = schemas.TASK_PROMPT[cls]
    tr = os.path.join(dst_root, "train").replace("\\", "/")
    va = os.path.join(dst_root, "val").replace("\\", "/")
    print("\n" + "=" * 70)
    print("아래를 donut_training.ipynb 의 Step 1 CFG 에 반영하세요 (로컬 모드):")
    print("=" * 70)
    print(f'''CFG["model"]["image_size"]   = {reco["image_size"]}   # 패치 크기에 맞춤(튜닝 가능)
CFG["model"]["max_length"]   = {reco["max_length"]}    # 라벨 JSON 이 짧아 작게
CFG["data"]["dataset_name"]  = None          # 로컬 모드
CFG["data"]["task_prompt"]   = "{task}"
CFG["data"]["local_train_dir"] = "{tr}"
CFG["data"]["local_val_dir"]   = "{va}"
# 학습 하이퍼(논문 P2 참고): num_epochs=30, learning_rate=3e-5, fp16=True
# 그리고 cell 9(로컬 데이터셋 준비)는 **건너뛰세요** — 이미 정리됨.''')
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="donut_anno → donut_vml 학습 포맷 변환")
    ap.add_argument("--src", default="datasets/donut_anno", help="donut_data 산출 루트")
    ap.add_argument("--dvml", default="/home/jhkim/projects/donut_vml", help="donut_vml 루트")
    ap.add_argument("--class", dest="cls", required=True, choices=schemas.CLASSES)
    ap.add_argument("--verified-only", action="store_true", help="manifest verified==1 만")
    args = ap.parse_args()

    verified = load_verified(args.src) if args.verified_only else None
    if args.verified_only and verified is None:
        sys.exit(f"manifest.csv 없음: {args.src} (검수 필터 불가)")

    dst_root = os.path.join(args.dvml, "data", "processed", f"drawing_{args.cls}")
    n_tr = convert_split(args.src, "train", args.cls, dst_root, verified)
    n_va = convert_split(args.src, "val",   args.cls, dst_root, verified)

    print(f"[변환] class={args.cls}  train {n_tr} / val {n_va}  → {dst_root}")
    if n_tr == 0:
        sys.exit("학습 샘플 0개 — 클래스/검수 필터/경로를 확인하세요.")
    if n_va == 0:
        print("  [주의] val 0개 — donut_data build 시 --split 의 VAL 비율을 늘리거나"
              " 데이터를 더 모으세요(평가가 불가).")
    print_cfg(args.cls, dst_root)


if __name__ == "__main__":
    main()
