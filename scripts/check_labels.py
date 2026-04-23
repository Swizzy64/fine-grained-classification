from src.dataset.annotations import load_annotations

samples = load_annotations(
    "data/raw/car_devkit/devkit/cars_train_annos.mat",
    "data/raw/cars_train/cars_train"
)

print("MIN LABEL:", min(y for _, y in samples))
print("MAX LABEL:", max(y for _, y in samples))