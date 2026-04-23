import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import time

from src.dataset.annotations import load_annotations
from src.dataset.splits import create_standard_split
from src.dataset.cars_dataset import CarsDataset
from src.models.resnet import get_resnet50
from src.training.trainer import train_one_epoch, evaluate

def main():
    samples = load_annotations(
        "data/raw/car_devkit/devkit/cars_train_annos.mat",
        "data/raw/cars_train/cars_train"
    )

    train, val, test = create_standard_split(samples)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_ds = CarsDataset(train, transform)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    val_ds = CarsDataset(val, transform)

    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_resnet50(num_classes=196).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(5):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"Execution time: {end_time - start_time:.2f} seconds"
    )

if __name__ == "__main__":
    main()