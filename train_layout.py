"""도면 레이아웃(title_block / view / note) YOLOv11-det 학습·검증·추론 스크립트.

§14·논문 P2의 1단계 검출기: 양식-의존 ROI 휴리스틱(collect_titleblock 기본)을 대체해
표제란/뷰/노트 영역을 검출한다. 영역은 수평 박스(AABB)라 OBB가 아닌 일반 det 사용.
(주석 치수/GD&T는 회전이 있어 OBB → train_obb.py 가 담당. 두 검출기는 별개.)

선행:
    /home/jhkim/anaconda3/envs/kardi_env/bin/pip install ultralytics --no-deps   # torch 보존
    # 데이터: datasets/drawing_layout/{images,labels}/{train,val}
    #   labels 는 det 포맷 (class cx cy w h, 0~1 정규화)
    # 라벨링: Roboflow / labelImg / X-AnyLabeling → "YOLO" 포맷 export
    #   클래스 순서 = drawing_layout.yaml names (0=title_block 1=view 2=note)

사용 예:
    python train_layout.py                                   # 학습
    python train_layout.py --model yolo11s.pt --epochs 150
    python train_layout.py --mode val
    python train_layout.py --mode predict --source <도면.png|폴더>
"""
import argparse
import os

PROJECT_DIR = "/home/jhkim/projects/Title_Qwen_VL_25_7B"
DATA_YAML = os.path.join(PROJECT_DIR, "drawing_layout.yaml")
RUN_PROJECT = os.path.join(PROJECT_DIR, "yolo_layout_drawing")  # 결과 저장 루트


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "val", "predict"], default="train")
    # 사전학습 det 가중치(OBB 아님). n<s<m<l<x 정확도↑/속도↓. 데이터 적으면 n/s.
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=100)
    # 표제란/뷰는 큰 영역이라 imgsz 1280 이면 충분(작은 글씨 검출이 목적이 아님).
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=float, default=-1)   # -1 = GPU 메모리 자동
    p.add_argument("--device", default="0")
    p.add_argument("--name", default="train")
    p.add_argument("--source", default=os.path.join(PROJECT_DIR, "input_doc"))
    p.add_argument("--weights", default=os.path.join(RUN_PROJECT, "train", "weights", "best.pt"))
    p.add_argument("--conf", type=float, default=0.25)
    return p.parse_args()


def main():
    a = parse_args()
    from ultralytics import YOLO

    if a.mode == "train":
        model = YOLO(a.model)               # 사전학습 det 가중치에서 전이학습
        model.train(
            data=DATA_YAML, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
            device=a.device, project=RUN_PROJECT, name=a.name,
            patience=30,
            # ── 레이아웃 특화 증강(보수적) ──────────────────────────────
            # 표제란/뷰는 위치가 의미(우하단 등) → 뒤집기 끔. 영역은 축정렬 → 회전 끔.
            degrees=0.0, fliplr=0.0, flipud=0.0,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,   # 흑백 선화 → 색 변형 무의미(밝기만)
            mosaic=0.5,
        )
        best = os.path.join(RUN_PROJECT, a.name, "weights", "best.pt")
        print(f"\n[완료] best.pt -> {best}")
        print("표제란 크롭에 쓰려면:")
        print(f"  python donut_data/collect_titleblock.py --input <도면> --weights {best} --out patches/titleblock")

    elif a.mode == "val":
        model = YOLO(a.weights)
        print(model.val(data=DATA_YAML, imgsz=a.imgsz, device=a.device,
                        project=RUN_PROJECT, name="val"))

    elif a.mode == "predict":
        model = YOLO(a.weights)
        model.predict(a.source, imgsz=a.imgsz, device=a.device, conf=a.conf,
                      save=True, project=RUN_PROJECT, name="predict")
        print(f"검출 결과 이미지 저장됨 -> {os.path.join(RUN_PROJECT, 'predict')}")


if __name__ == "__main__":
    main()
