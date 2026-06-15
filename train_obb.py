"""도면 주석(measure / gdt / radii) YOLO-OBB 학습·검증·추론 스크립트.

선행 조건:
    pip install ultralytics
    # 데이터 준비: datasets/drawing_obb/{images,labels}/{train,val}
    #   labels 는 OBB 포맷 (class x1 y1 x2 y2 x3 y3 x4 y4, 0~1 정규화)
    # 라벨링 도구 예: roboflow, labelImg(OBB), X-AnyLabeling

사용 예:
    python train_obb.py                 # 학습
    python train_obb.py --model yolo11s-obb.pt --epochs 150
    python train_obb.py --mode val      # 검증
    python train_obb.py --mode predict --source doc_sample_paper.jpg
"""
import argparse
import os

from ultralytics import YOLO

PROJECT_DIR = "/home/jhkim/projects/Title_Qwen_VL_25_7B"
DATA_YAML = os.path.join(PROJECT_DIR, "drawing_obb.yaml")
RUN_PROJECT = os.path.join(PROJECT_DIR, "yolo_obb_drawing")  # 결과 저장 루트


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "val", "predict"], default="train")
    # 사전학습 OBB 가중치. n<s<m<l<x 순으로 정확도↑/속도↓. 데이터 적으면 n/s 권장.
    p.add_argument("--model", default="yolo11n-obb.pt")
    p.add_argument("--epochs", type=int, default=100)
    # 도면 글자가 작아 고해상도가 중요. 메모리 부족 시 960으로 낮출 것.
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=float, default=-1)   # -1 = GPU 메모리에 맞춰 자동
    p.add_argument("--device", default="0")             # GPU 0
    p.add_argument("--name", default="train")           # 실행 폴더 이름
    p.add_argument("--source", default=os.path.join(PROJECT_DIR, "doc_sample_paper.jpg"))
    p.add_argument("--weights", default=os.path.join(RUN_PROJECT, "train", "weights", "best.pt"))
    return p.parse_args()


def main():
    a = parse_args()

    if a.mode == "train":
        model = YOLO(a.model)               # 사전학습 가중치에서 시작(전이학습)
        model.train(
            data=DATA_YAML,
            epochs=a.epochs,
            imgsz=a.imgsz,
            batch=a.batch,
            device=a.device,
            project=RUN_PROJECT,
            name=a.name,
            patience=30,                    # 30 에폭 개선 없으면 조기종료
            # ── 도면 특화 증강 설정 ──────────────────────────────────────
            # 도면은 색/뒤집힘 변형이 의미를 해칠 수 있어 보수적으로.
            degrees=10.0,                   # OBB라 약간의 회전 증강은 유익
            fliplr=0.0,                     # 좌우반전 끔(R/숫자 방향 보존)
            flipud=0.0,                     # 상하반전 끔
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,  # 색상 변형 최소(밝기만 약간)
            mosaic=0.5,
        )
        # 학습 후 best.pt 위치 안내 → 노트북 OBB_WEIGHTS 에 지정.
        best = os.path.join(RUN_PROJECT, a.name, "weights", "best.pt")
        print(f"\n[완료] best.pt -> {best}")
        print("노트북 셀 7-1 의 OBB_WEIGHTS 를 위 경로로 지정하세요.")

    elif a.mode == "val":
        model = YOLO(a.weights)
        metrics = model.val(data=DATA_YAML, imgsz=a.imgsz, device=a.device,
                            project=RUN_PROJECT, name="val")
        print(metrics)

    elif a.mode == "predict":
        model = YOLO(a.weights)
        model.predict(a.source, imgsz=a.imgsz, device=a.device, conf=0.25,
                      save=True, project=RUN_PROJECT, name="predict")
        print(f"검출 결과 이미지 저장됨 -> {os.path.join(RUN_PROJECT, 'predict')}")


if __name__ == "__main__":
    main()
