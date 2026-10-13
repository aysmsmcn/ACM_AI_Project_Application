
import os

from ultralytics import YOLO


WEIGHTS = r"C:\Users\aysms\Documents\Python Scripts\Computer Vision Project\src\runs\segment\cloud_n_416\weights\best.pt"
SOURCE = r"C:\Users\aysms\Documents\Python Scripts\Computer Vision Project\data\processed\images\test"
CONF = 0.25
IMGSZ = 416

def require_file(path, setting_name):
    if not os.path.exists(path):
        raise SystemExit(
            "Cannot find " + path + "\n"
            "Set " + setting_name + " at the top of this script to the FULL path of that file or folder.\n"
            "On Windows it looks like r\"C:\\Users\\you\\...\" (with the r before the quotes).\n"
            "This script is running from: " + os.getcwd()
        )


def main():
    require_file(WEIGHTS, "WEIGHTS")
    require_file(SOURCE, "SOURCE")
    model = YOLO(WEIGHTS)
    results = model.predict(
        source=SOURCE,
        conf=CONF,
        imgsz=IMGSZ,
        save=True,
        stream=True,
        name="predict_test",
        exist_ok=True,
    )

    for i, result in enumerate(results):
        found = [model.names[int(c)] for c in result.boxes.cls] if result.boxes is not None else []
        print(f"{result.path}: {len(found)} regions {sorted(set(found))}")

    print("Saved pictures are in the newest folder under runs/segment/ (predict_test)")


if __name__ == "__main__":
    main()