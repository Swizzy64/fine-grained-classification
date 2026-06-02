import torch
from pathlib import Path


CHECKPOINT_DIR = Path("outputs")

def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)

def summarize_all_checkpoints():
    print("\nSaved checkpoints")
    print("-" * 60)
    print(f"{'Model':<15}{'Epoch':<10}{'Val Acc':<15}")
    print("-" * 60)

    checkpoints = sorted(CHECKPOINT_DIR.glob("best_*.pth"))

    all_checkpoints = []

    for ckpt_path in checkpoints:
        ckpt = load_checkpoint(ckpt_path)

        model_name = ckpt.get("model_name", ckpt_path.stem)
        epoch = ckpt.get("epoch", -1)
        val_acc = ckpt.get("val_acc", -1)

        all_checkpoints.append((model_name, epoch, val_acc, ckpt_path))

        print(f"{model_name:<15}{epoch:<10}{val_acc:<15.4f}")

    return all_checkpoints

def inspect_checkpoint(path):
    ckpt = load_checkpoint(path)

    print("\nDetailed inspection")
    print("-" * 60)
    print(f"File: {path.name}")
    print(f"Model: {ckpt.get('model_name')}")
    print(f"Best epoch: {ckpt.get('epoch')}")
    print(f"Validation accuracy: {ckpt.get('val_acc'):.4f}")

    print("\nFirst 10 stored parameters:")
    state_dict = ckpt["model_state"]

    for i, (k, v) in enumerate(state_dict.items()):
        if i >= 10:
            break
        print(f"{k:<45} {list(v.shape) if hasattr(v, 'shape') else v}")

def main():
    ckpts = summarize_all_checkpoints()

    for _, _, _, path in ckpts:
        inspect_checkpoint(path)

if __name__ == "__main__":
    main()