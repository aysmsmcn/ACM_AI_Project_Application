# ACM AI Project Application

I built a program that detects different types of clouds.

I chose to finetune a model using the dataset Clouds-1500 from Mendeley Data: <https://data.mendeley.com/datasets/2khchjbgzr/2>

I also chose to use the YOLO model from Ultralytics because I played around with it a little over the summer

All libraries used are listed in requirements.txt

What I did:
- First I converted the annotations in the dataset to YOLO formatted labels using the script convert.py
- Then I trained the YOLO26-seg model in the finetuning.py script
  - I first ran a smoke test then the full training
- Results are saved in the runs folder
- I proceeded to evaluate the results



### 5. Evaluate

```
python scripts/evaluate.py
```

Prints mask mAP overall and per class. Use `SPLIT = "val"` while experimenting. Switch
to `"test"` only once, at the very end, so the test score stays honest.

### 6. Look at predictions

```
python scripts/predict.py
```

Saves pictures with the predicted masks drawn on. Compare them with the real masks to
see what the model gets wrong.

### 7. Experiment

Change one setting at a time in `train.py`, give each run its own `RUN_NAME`, and
compare on the val split. Ideas:

- a larger model size (`n`, `s`, `m`)
- a larger `IMGSZ` (photos are about 2,500 pixels wide, so thin cloud streaks can vanish at 640)
- `freeze=10` in the `model.train()` call, which can reduce overfitting
- `mosaic=0`, since mosaic stitches four photos into one and skies rarely look like that

## Decisions made for this prototype

- **Random split within each camera.** Every photo was taken on a different day, so
  near-identical neighboring frames are not a concern, and no date table is needed.
- **No coverage table.** The per-class counts Ultralytics prints at validation show
  whether a class is missing from a split.
- **Polygon rescaling skipped.** Normalized YOLO labels handle the size difference.
- **Burned-in text left in.** It is a shortcut risk mainly for classification or when a
  whole camera is held out. If predicted masks show odd blobs in the bottom corners, fill
  the corner boxes with `cv2.rectangle` and re-run the split on the cleaned copies.
- **No baselines.** A most-common-class guess, a YOLO model trained from scratch, and an
  OpenCV red/blue color threshold would each give a reference point, but none is needed
  to get a working model. A from-scratch run is the most instructive one to add later.

## Known limitations

- YOLO segmentation is instance-based, and clouds are not discrete objects. A large layer
  may come out as several overlapping pieces. To get per-pixel scores such as mIoU, merge
  the predicted pieces into one class map, resize it to the mask size with
  nearest-neighbor, and compare it with `masks_machine`. That script is not written yet.
- The `Arvore` class inflates average scores because it is large and always in the same
  place for a given camera.
- Mask mAP from `evaluate.py` is not the same measure as per-pixel mIoU, so do not
  compare the two directly.
- The scripts were written without being run on the full dataset. Folder names such as
  `img` and `ann` are guesses to check against your download.

## Troubleshooting

- **"System cannot find the path specified" or "Cannot find ...":** a path in the edit
  block is wrong, or you are running from a different folder than you think. Use full
  paths. On Windows, write `r"C:\Users\you\..."` with the `r` before the quotes.
- **`dataset.yaml` is found but images are not:** open it and check that its `path:` line
  points to the real `dataset` folder. It was written when you ran `convert_labels.py`.
- **`torch.cuda.is_available()` prints `False` even though you have an NVIDIA GPU:** pip
  sometimes installs a CPU-only PyTorch, especially on Windows. Install the CUDA build
  using the selector on <https://pytorch.org>, then check again.
- **Out of GPU memory:** lower `BATCH` in `train.py` to 8 or 4, then lower `IMGSZ`.
- **A crash right at the start of training on Windows:** keep the
  `if __name__ == "__main__":` guard that the scripts already have.
- **A missing-annotation error from `convert_labels.py`:** it prints the file names it
  looked for. Compare them with the real names in your annotations folder.

## Results

Fill this in as you run experiments.

| Run name | Model | Image size | Val mask mAP50 | Val mask mAP50-95 | Notes |
|---|---|---|---|---|---|
| cloud_n_640 | yolo26n-seg | 640 | | | first real run |

Test split (scored once, at the end):

| Model | Test mask mAP50 | Test mask mAP50-95 |
|---|---|---|
| | | |

## Licenses

Check the license terms of the dataset on its Mendeley page. As far as I know,
Ultralytics is released under AGPL-3.0, so check its terms too before using this beyond
a personal or course project.
