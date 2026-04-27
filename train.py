import argparse
import json
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset_preparation import (
    get_num_classes,
    get_split1_loaders,
    get_split2_loaders,
    get_split3_loaders,
)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path("training_results")
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")

IS_CUDA = DEVICE.type == "cuda"
TRAIN_IMG_SIZE = 224 if IS_CUDA else 128  
NUM_WORKERS = 4 if IS_CUDA else 0         
BATCH_SIZE = 32 if IS_CUDA else 16        


def create_model(num_classes: int, pretrained: bool = True, drop_rate: float = 0.2):
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer,
        scheduler=None,
        device=DEVICE,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epoch_time": [],
        }

    def train_one_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc="  Train", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).long()

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc="  Val  ", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).long()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        save_path: str,
        patience: int = 0,
    ):
       
        best_val_acc = -1.0
        best_epoch = 0
        epochs_no_improve = 0
        total_train_time = 0.0

        print(f"\n{'='*60}")
        print(f"Training for {epochs} epochs -> {save_path}")
        print(f"{'='*60}")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            epoch_time = time.time() - epoch_start
            total_train_time += epoch_time

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["epoch_time"].append(epoch_time)

            # Print epoch summary
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
                f"LR: {lr:.2e} | Time: {epoch_time:.1f}s"
            )

            # Save best model (by validation accuracy)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "train_loss": train_loss,
                    },
                    save_path,
                )
                print(f"  [*] Saved best model (val_acc={val_acc:.4f})")
            else:
                epochs_no_improve += 1

            # Early stopping
            if patience > 0 and epochs_no_improve >= patience:
                print(f"\n  Early stopping after {patience} epochs without improvement")
                break

        print(f"\n{'-'*60}")
        print(f"Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        print(f"Total training time: {total_train_time:.1f}s")
        print(f"{'-'*60}")

        self.history["best_val_acc"] = best_val_acc
        self.history["best_epoch"] = best_epoch
        self.history["total_time"] = total_train_time

        return self.history




def run_experiment_1():
    print("\n" + "#" * 60)
    print("  EXPERIMENT 1 -- SPLIT1 (raw, 5% of train)")
    print("#" * 60)

    num_classes = get_num_classes()
    epochs = 10 if not IS_CUDA else 20

    loader_train, loader_val, _ = get_split1_loaders(
        batch_size=BATCH_SIZE, fraction=0.05, num_workers=NUM_WORKERS,
        img_size=TRAIN_IMG_SIZE,
    )

    model = create_model(num_classes, pretrained=True, drop_rate=0.2)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    trainer = Trainer(model, criterion, optimizer, scheduler)
    history = trainer.fit(
        loader_train, loader_val,
        epochs=epochs,
        save_path=str(MODELS_DIR / "MODEL1.pth"),
    )

    # Save training history
    save_history(history, RESULTS_DIR / "history_split1.json")
    return history


def run_overfit_experiment():
    print("\n" + "#" * 60)
    print("  OVERFITTING EXPERIMENT -- SPLIT1 (raw, 5%, no dropout)")
    print("#" * 60)

    num_classes = get_num_classes()
    epochs = 10 if not IS_CUDA else 20

    loader_train, loader_val, _ = get_split1_loaders(
        batch_size=BATCH_SIZE, fraction=0.05, num_workers=NUM_WORKERS,
        img_size=TRAIN_IMG_SIZE,
    )

    # No dropout to encourage overfitting
    model = create_model(num_classes, pretrained=True, drop_rate=0.0)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)  # No weight decay

    trainer = Trainer(model, criterion, optimizer, scheduler=None)
    history = trainer.fit(
        loader_train, loader_val,
        epochs=epochs,
        save_path=str(MODELS_DIR / "MODEL_OVERFIT.pth"),
    )

    save_history(history, RESULTS_DIR / "history_overfit.json")
    return history


def run_experiment_2():
    print("\n" + "#" * 60)
    print("  EXPERIMENT 2 -- SPLIT2 (normalised + augmented, full train)")
    print("#" * 60)

    num_classes = get_num_classes()
    epochs = 10 if not IS_CUDA else 20

    loader_train, loader_val, _ = get_split2_loaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, img_size=TRAIN_IMG_SIZE,
    )

    model = create_model(num_classes, pretrained=True, drop_rate=0.2)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    trainer = Trainer(model, criterion, optimizer, scheduler)
    history = trainer.fit(
        loader_train, loader_val,
        epochs=epochs,
        save_path=str(MODELS_DIR / "MODEL2.pth"),
    )

    save_history(history, RESULTS_DIR / "history_split2.json")
    return history


def run_experiment_3():
    print("\n" + "#" * 60)
    print("  EXPERIMENT 3 -- SPLIT3 (val from train pool)")
    print("#" * 60)

    num_classes = get_num_classes()
    epochs = 10 if not IS_CUDA else 20

    loader_train, loader_val, _ = get_split3_loaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, img_size=TRAIN_IMG_SIZE,
    )

    model = create_model(num_classes, pretrained=True, drop_rate=0.2)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    trainer = Trainer(model, criterion, optimizer, scheduler)
    history = trainer.fit(
        loader_train, loader_val,
        epochs=epochs,
        save_path=str(MODELS_DIR / "MODEL3.pth"),
    )

    save_history(history, RESULTS_DIR / "history_split3.json")
    return history


def save_history(history: dict, path: Path):
    serializable = {}
    for k, v in history.items():
        if isinstance(v, (list, tuple)):
            serializable[k] = [float(x) if isinstance(x, (int, float, np.floating)) else x for x in v]
        elif isinstance(v, (int, float, np.floating)):
            serializable[k] = float(v)
        else:
            serializable[k] = v

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  History saved to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3: ML Model Training")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "1", "2", "3", "overfit"],
        help="Which experiment to run (default: all)",
    )
    args = parser.parse_args()

    experiments = {
        "1": run_experiment_1,
        "overfit": run_overfit_experiment,
        "2": run_experiment_2,
        "3": run_experiment_3,
    }

    if args.experiment == "all":
        for name, func in experiments.items():
            func()
    else:
        experiments[args.experiment]()

    print("\n\n" + "=" * 60)
    print("  ALL DONE -- run plot_results.py to generate plots")
    print("=" * 60)
