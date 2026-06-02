import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import time
from pathlib import Path
import copy

from src.dataset.annotations import load_annotations
from src.dataset.splits import create_standard_split
from src.dataset.cars_dataset import CarsDataset
from src.models.resnet import get_resnet50
from src.models.efficientnet import get_efficientnet_b3
from src.models.vit import get_vit_b16
from src.training.trainer import train_one_epoch, evaluate


def get_model(name, num_classes, device):
    if name == "resnet":
        model = get_resnet50(num_classes=num_classes)
    elif name == "efficientnet":
        model = get_efficientnet_b3(num_classes=num_classes)
    elif name == "vit":
        model = get_vit_b16(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")

    return model.to(device)

def create_logger(path):
    path = Path(path)
    path.parent.mkdir(exist_ok=True)

    def log(text):
        print(text)
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    return log

def save_checkpoint(path, model_state, optimizer_state, epoch, val_acc, model_name):
    torch.save({
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "epoch": epoch,
        "val_acc": val_acc,
        "model_name": model_name
    }, path)

def main():
    samples = load_annotations(
        "data/raw/car_devkit/devkit/cars_train_annos.mat",
        "data/raw/cars_train/cars_train"
    )

    train, val, test = create_standard_split(samples)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_loader = DataLoader(
        CarsDataset(train, transform),
        batch_size=8,
        shuffle=True
    )

    val_loader = DataLoader(
        CarsDataset(val, transform),
        batch_size=8,
        shuffle=False
    )

    test_loader = DataLoader(
        CarsDataset(test, transform),
        batch_size=8,   
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = create_logger("outputs/experiment_log.txt")

    models = ["resnet", "efficientnet", "vit"]
    num_epochs = 300

    lr = 1e-4
    weight_decay = 1e-4

    for model_name in models:
        log(f"Training {model_name.upper()}")

        if device.type == "cuda":
            log(f"Device: {torch.cuda.get_device_name(0)}")
        else:
            log("Device: CPU")

        model = get_model(model_name, num_classes=196, device=device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs
        )

        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(num_epochs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.time()

            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, device)

            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                best_epoch = epoch

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()
            
            log(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | "
                f"Execution time: {end_time - start_time:.2f} seconds"
            )

        log(f"Best validation accuracy for {model_name.upper()}: {best_val_acc:.4f}")

        if best_model_state is not None:
            save_checkpoint(
                f"outputs/best_{model_name}.pth",
                best_model_state,
                best_optimizer_state,
                best_epoch,
                best_val_acc,
                model_name
            ) 

        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()