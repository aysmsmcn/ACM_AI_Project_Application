
import os

from ultralytics import YOLO

WEIGHTS = r"C:\Users\aysms\Documents\Python Scripts\Computer Vision Project\src\runs\segment\cloud_n_416\weights\best.pt"
DATA = r"C:\Users\aysms\Documents\Python Scripts\Computer Vision Project\data\processed\dataset.yaml"
SPLIT = "test"
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
    require_file(DATA, "DATA")
    model = YOLO(WEIGHTS)
    metrics = model.val(data=DATA, split=SPLIT, imgsz=IMGSZ, plots=True)

    print(f"\nResults on the {SPLIT} split")
    print(f"Mask mAP50:     {metrics.seg.map50:.3f}")
    print(f"Mask mAP50-95:  {metrics.seg.map:.3f}")

    names = model.names
    per_class = metrics.seg.maps
    print("\nMask mAP50-95 per class:")
    for class_id, name in names.items():
        print(f"  {name}: {per_class[class_id]:.3f}")



if __name__ == "__main__":
    main()