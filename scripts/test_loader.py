import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision

from src.dataset.annotations import load_annotations
from src.dataset.cars_dataset import CarsDataset


def show_batch(images, labels):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0)
        ax.imshow(img)
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    train_mat_file = "data/raw/car_devkit/devkit/cars_train_annos.mat"
    train_img_dir = "data/raw/cars_train/cars_train"

    train_samples = load_annotations(train_mat_file, train_img_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = CarsDataset(train_samples, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2
    )

    images, labels = next(iter(dataloader))

    print("Images shape:", images.shape)
    print("Labels:", labels)

    show_batch(images, labels)


if __name__ == "__main__":
    main()