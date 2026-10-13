from ultralytics import YOLO
import torch

SMOKE_TEST = False
MODEL = "yolo26n-seg.pt"
DATA = r"C:\Users\aysms\Documents\Python Scripts\Computer Vision Project\data\processed\dataset.yaml"
IMGSZ = 416
EPOCHS = 30
PATIENCE = 20
BATCH = 8
RUN_NAME = "cloud_n_416"
SEED = 0


def main():
    print("GPU available:", torch.cuda.is_available())

    if SMOKE_TEST:
        run_name, epochs, fraction = "smoke", 3, 0.1
    else:
        run_name, epochs, fraction = RUN_NAME, EPOCHS, 1.0

    model = YOLO(MODEL)
    model.train(
        data=DATA,
        epochs=epochs,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        fraction=fraction,
        seed=SEED,
        name=run_name,
        exist_ok=SMOKE_TEST,
        plots=True,
        cache="ram",
    )
    print("Done. Results are in:", model.trainer.save_dir)
    print("Best model:", model.trainer.best)

if __name__ == "__main__":
    main()