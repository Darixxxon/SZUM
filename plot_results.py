import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.figsize": (8, 6),
    }
)

RESULTS_DIR = Path("training_results")
PLOTS_DIR = Path("training_plots")
PLOTS_DIR.mkdir(exist_ok=True)

COLORS = {
    "train": "#2196F3",     
    "val": "#FF5722",
    "train_fill": "#90CAF9",
    "val_fill": "#FFAB91",
}


def load_history(name: str) -> dict:
    path = RESULTS_DIR / f"history_{name}.json"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def plot_loss_and_accuracy(history, title, filename_prefix, highlight_best=True):
    if history is None:
        return

    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Loss ──
    plt.figure()
    plt.plot(epochs, history["train_loss"], color=COLORS["train"], linewidth=2, label="Train Loss", marker="o", markersize=3)
    plt.plot(epochs, history["val_loss"], color=COLORS["val"], linewidth=2, label="Val Loss", marker="s", markersize=3)
    plt.fill_between(epochs, history["train_loss"], alpha=0.1, color=COLORS["train_fill"])
    plt.fill_between(epochs, history["val_loss"], alpha=0.1, color=COLORS["val_fill"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss (CrossEntropy)")
    plt.title(f"{title} — Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if len(epochs) == 1:
        plt.xlim(0.5, 1.5)
        plt.xticks([1])
    
    plt.tight_layout()
    save_path = PLOTS_DIR / f"{filename_prefix}_loss.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

    # ── Accuracy ──
    plt.figure()
    plt.plot(epochs, history["train_acc"], color=COLORS["train"], linewidth=2, label="Train Accuracy", marker="o", markersize=3)
    plt.plot(epochs, history["val_acc"], color=COLORS["val"], linewidth=2, label="Val Accuracy", marker="s", markersize=3)
    plt.fill_between(epochs, history["train_acc"], alpha=0.1, color=COLORS["train_fill"])
    plt.fill_between(epochs, history["val_acc"], alpha=0.1, color=COLORS["val_fill"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} — Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    if len(epochs) == 1:
        plt.xlim(0.5, 1.5)
        plt.xticks([1])

    plt.tight_layout()
    save_path = PLOTS_DIR / f"{filename_prefix}_accuracy.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")





def plot_comparison(histories, filename_prefix="comparison"):
    labels = []
    val_accs = []
    train_accs = []
    times = []
    colors_bar = ["#42A5F5", "#66BB6A", "#FFA726"]

    for name, label, color in [
        ("split1", "SPLIT1\n(5% raw)", "#42A5F5"),
        ("split2", "SPLIT2\n(full + augment)", "#66BB6A"),
        ("split3", "SPLIT3\n(val in train)", "#FFA726"),
        ("overfit", "OVERFIT\n(5% raw)", "#EF5350"),
    ]:
        h = histories.get(name)
        if h is None:
            continue
        labels.append(label)
        val_accs.append(h.get("best_val_acc", 0))
        train_accs.append(h["train_acc"][-1] if h["train_acc"] else 0)
        times.append(h.get("total_time", 0))
        colors_bar.append(color)

    if not labels:
        print("  [SKIP] No histories available for comparison")
        return

    # ── Accuracy comparison ──
    plt.figure()
    x = np.arange(len(labels))
    width = 0.35
    bars1 = plt.bar(x - width / 2, train_accs, width, label="Train Acc", color="#90CAF9", edgecolor="#1565C0")
    bars2 = plt.bar(x + width / 2, val_accs, width, label="Best Val Acc", color="#A5D6A7", edgecolor="#2E7D32")

    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison Across Splits")
    plt.xticks(x, labels)
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_path = PLOTS_DIR / f"{filename_prefix}_accuracy.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

    # ── Time comparison ──
    plt.figure()
    bars3 = plt.bar(labels, times, color=["#42A5F5", "#66BB6A", "#FFA726", "#EF5350"][:len(labels)], edgecolor="gray")
    plt.ylabel("Total Training Time (s)")
    plt.title("Training Time Comparison")
    plt.grid(True, alpha=0.3, axis="y")

    for bar in bars3:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h + 1, f"{h:.0f}s", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_path = PLOTS_DIR / f"{filename_prefix}_time.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_all_val_curves(histories, filename="val_curves_all.png"):
    plt.figure()

    style = [
        ("split1", "SPLIT1 (5% raw)", "#2196F3", "-"),
        ("split2", "SPLIT2 (full + augment)", "#4CAF50", "-"),
        ("split3", "SPLIT3 (val in train)", "#FF9800", "-"),
        ("overfit", "OVERFIT (5% raw)", "#F44336", "-"),
    ]

    for name, label, color, ls in style:
        h = histories.get(name)
        if h is None:
            continue
        epochs = range(1, len(h["val_acc"]) + 1)
        plt.plot(epochs, h["val_acc"], color=color, linewidth=2.5, label=label, linestyle=ls, marker="o", markersize=4)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Across All Splits")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    plt.tight_layout()
    save_path = PLOTS_DIR / filename
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

if __name__ == "__main__":
    print("Generating training plots...")

    h_split1 = load_history("split1")
    h_overfit = load_history("overfit")
    h_split2 = load_history("split2")
    h_split3 = load_history("split3")

    histories = {}
    if h_split1:
        histories["split1"] = h_split1
    if h_split2:
        histories["split2"] = h_split2
    if h_split3:
        histories["split3"] = h_split3
    if h_overfit:
        histories["overfit"] = h_overfit

    # Individual split plots
    plot_loss_and_accuracy(h_split1, "SPLIT1 — 5% Raw Data Training", "split1_training")
    plot_loss_and_accuracy(h_split2, "SPLIT2 — Full Augmented Training", "split2_training")
    plot_loss_and_accuracy(h_split3, "SPLIT3 — Val from Train Pool", "split3_training")
    plot_loss_and_accuracy(h_overfit, "Overfitting Experiment\n(5% data, no dropout, no weight decay)", "overfitting")

    if len(histories) >= 2:
        plot_comparison(histories)
        plot_all_val_curves(histories)

    print("\nAll plots saved to:", PLOTS_DIR)
