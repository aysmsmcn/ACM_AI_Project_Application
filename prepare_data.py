
import csv
import random
import shutil
from pathlib import Path

RAW_ROOT = Path(r"C:\Users\aysms\Documents\Python Scripts\Computer Vision Project\data\raw")
PHOTOS_SUBDIR = "img"
OUT_ROOT = Path(r"C:\Users\aysms\Documents\Python Scripts\Computer Vision Project\data\processed")
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 0

random.seed(SEED)
rows = []


for camera_dir in sorted(p for p in RAW_ROOT.iterdir() if p.is_dir()):
    photos = sorted((camera_dir / PHOTOS_SUBDIR).glob("*.jpg"))
    random.shuffle(photos)

    n = len(photos)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])
    groups = {
        "train": photos[:n_train],
        "val": photos[n_train:n_train + n_val],
        "test": photos[n_train + n_val:],
    }

    for split, files in groups.items():
        out_dir = OUT_ROOT / "images" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:

            new_name = f"{camera_dir.name}_{f.name}"
            shutil.copy2(f, out_dir / new_name)
            rows.append([camera_dir.name, str(f), split, new_name])

    counts = ", ".join(f"{k} {len(v)}" for k, v in groups.items())
    print(f"{camera_dir.name}: {n} photos -> {counts}")

with open(OUT_ROOT / "split.csv", "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["camera", "original_path", "split", "new_name"])
    writer.writerows(rows)

print(f"Done. Wrote {len(rows)} photos and {OUT_ROOT / 'split.csv'}")