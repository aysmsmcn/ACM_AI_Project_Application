#converts the dataset's annotations to YOLO segmentation labels
import csv
import json
import os
from pathlib import Path


RAW_ROOT = Path(r"C:\Users\aysms\Documents\Python Scripts\Computer Vision Project\data\raw")
ANN_SUBDIR = "ann"
OUT_ROOT = Path(r"C:\Users\aysms\Documents\Python Scripts\Computer Vision Project\data\processed")


def find_annotation(camera, photo_name):
    stem = os.path.splitext(photo_name)[0]
    ann_dir = os.path.join(RAW_ROOT, camera, ANN_SUBDIR)
    candidates = [
        os.path.join(ann_dir, stem + "_jpg.json"),
        os.path.join(ann_dir, photo_name + ".json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "No annotation found for " + photo_name + ". Looked for:\n  " + "\n  ".join(candidates)
    )


with open(os.path.join(OUT_ROOT, "split.csv"), newline="") as fh:
    rows = list(csv.DictReader(fh))

items = []
class_names = set()
skipped_shapes = 0
skipped_tiny = 0

for row in rows:
    photo_name = os.path.basename(row["original_path"])
    with open(find_annotation(row["camera"], photo_name), encoding="utf-8") as f:
        ann = json.load(f)

    width, height = ann["size"]["width"], ann["size"]["height"]
    polygons = []
    for obj in ann["objects"]:
        if obj.get("geometryType") != "polygon":
            skipped_shapes += 1
            continue
        points = obj["points"]["exterior"]
        if len(points) < 3:
            skipped_tiny += 1
            continue
        class_names.add(obj["classTitle"])
        polygons.append((obj["classTitle"], points))
    items.append((row, width, height, polygons))


class_ids = {name: i for i, name in enumerate(sorted(class_names))}

instance_counts = {name: 0 for name in class_ids}
empty_photos = 0

for row, width, height, polygons in items:
    label_dir = os.path.join(OUT_ROOT, "labels", row["split"])
    os.makedirs(label_dir, exist_ok=True)
    label_name = os.path.splitext(row["new_name"])[0] + ".txt"

    lines = []
    for class_title, points in polygons:
        numbers = []
        for x, y in points:
            numbers.append(f"{min(max(x / width, 0.0), 1.0):.6f}")
            numbers.append(f"{min(max(y / height, 0.0), 1.0):.6f}")
        lines.append(" ".join([str(class_ids[class_title])] + numbers))
        instance_counts[class_title] += 1

    if not lines:
        empty_photos += 1
    with open(os.path.join(label_dir, label_name), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

yaml_path = os.path.join(OUT_ROOT, "dataset.yaml")
with open(yaml_path, "w") as f:
    f.write("path: " + os.path.abspath(OUT_ROOT).replace("\\", "/") + "\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n")
    f.write("names:\n")
    for name, class_id in class_ids.items():
        f.write(f'  {class_id}: "{name}"\n')

print(f"Wrote {len(items)} label files and {yaml_path}")
print("Classes found (id: name, number of polygons):")
for name, class_id in class_ids.items():
    print(f"  {class_id}: {name}, {instance_counts[name]}")
print(f"Photos with no polygons: {empty_photos}")
if skipped_shapes or skipped_tiny:
    print(f"Skipped: {skipped_shapes} non-polygon shapes, {skipped_tiny} polygons with fewer than 3 points")