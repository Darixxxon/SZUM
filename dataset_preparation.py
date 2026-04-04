import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PLOT_DIR = Path("Plots")
PLOT_DIR.mkdir(exist_ok=True)
DATASET_ROOT = Path("Dataset")
IMG_SIZE     = 224
BATCH_SIZE   = 32
RANDOM_STATE = 42 # Set value for RANDOM_STATE which is used in train_test_split ensures reproducible data split

def collect_samples(root: Path):
    paths, labels = [], []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.glob("*.jpg"):
            paths.append(img_path)
            labels.append(class_dir.name)
    return np.array(paths), np.array(labels)

all_paths, all_labels = collect_samples(DATASET_ROOT)

# Encode string labels to integers, for example apple___healthy could be 0
le = LabelEncoder()
all_labels_enc = le.fit_transform(all_labels)

# Printing results of sampling
species  = sorted({l.split("___")[0] for l in le.classes_})
diseases = sorted({l.split("___")[1] for l in le.classes_})
print(f"Classes  : {len(le.classes_)}")
print(f"Species  : {species}")
print(f"Diseases : {diseases}")
print(f"Total    : {len(all_paths)} images")


# DATA SPLITTING - simple split with stratification for even class representation
# In all splits we assume fixed size of subsets - 80% train data, 10% val data and 10% test data
idx = np.arange(len(all_paths))

idx_train, idx_temp = train_test_split(
    idx, test_size=0.2, stratify=all_labels_enc, random_state=RANDOM_STATE
)
# 50% of 20% is 10% each - just for clarification
idx_val, idx_test = train_test_split(
    idx_temp, test_size=0.5, stratify=all_labels_enc[idx_temp], random_state=RANDOM_STATE
)

# Verify base split
print(f"\nBase split: \nTrain: {len(idx_train)}\nVal:{len(idx_val)}\nTest:{len(idx_test)}")

class LeafDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]

# SPLIT 1 — Raw  (only resize + tensor conversion, no normalisation)
raw_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

split1_full = LeafDataset(all_paths, all_labels_enc, transform=raw_transform)
split1_train = Subset(split1_full, idx_train)
split1_val   = Subset(split1_full, idx_val)
split1_test  = Subset(split1_full, idx_test)

loader1_train = DataLoader(split1_train, batch_size=BATCH_SIZE, shuffle=True)
loader1_val   = DataLoader(split1_val,   batch_size=BATCH_SIZE)
loader1_test  = DataLoader(split1_test,  batch_size=BATCH_SIZE)

print("\n### Split 1 (raw) ###")
print(f"Train : {len(split1_train)}\nVal : {len(split1_val)}\nTest : {len(split1_test)}")

# SPLIT 2 — Normalised + augmented
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# TRAIN — heavy augmentation to improve generalisation
# Suggested augmentations:
# - horizontal/vertical flipping
# - rotation
# - differentation in color distribution
# - zoomed/distanced photos
# - blurring
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# VAL / TEST — only resize + normalise (no augmentation, deterministic)
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

split2_train = LeafDataset(all_paths[idx_train], all_labels_enc[idx_train], train_transform)
split2_val   = LeafDataset(all_paths[idx_val],   all_labels_enc[idx_val],   eval_transform)
split2_test  = LeafDataset(all_paths[idx_test],  all_labels_enc[idx_test],  eval_transform)

loader2_train = DataLoader(split2_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
loader2_val   = DataLoader(split2_val,   batch_size=BATCH_SIZE, num_workers=4)
loader2_test  = DataLoader(split2_test,  batch_size=BATCH_SIZE, num_workers=4)

print("\n### Split 2 (normalised + augmented) ###")
print(f"Train : {len(split2_train)}\nVal : {len(split2_val)}\nTest : {len(split2_test)}")


# SPLIT 3 — VAL is a subset of the train pool
idx_pool, idx_test3 = train_test_split(
    idx, test_size=0.1, stratify=all_labels_enc, random_state=RANDOM_STATE
)
# VAL drawn from inside the pool (same 10% absolute size)
idx_train3, idx_val3 = train_test_split(
    idx_pool, test_size=len(idx_val) / len(idx_pool),
    stratify=all_labels_enc[idx_pool], random_state=RANDOM_STATE
)

split3_train = LeafDataset(all_paths[idx_train3], all_labels_enc[idx_train3], train_transform)
split3_val   = LeafDataset(all_paths[idx_val3],   all_labels_enc[idx_val3],   eval_transform)
split3_test  = LeafDataset(all_paths[idx_test3],  all_labels_enc[idx_test3],  eval_transform)

loader3_train = DataLoader(split3_train, batch_size=BATCH_SIZE, shuffle=True)
loader3_val   = DataLoader(split3_val,   batch_size=BATCH_SIZE)
loader3_test  = DataLoader(split3_test,  batch_size=BATCH_SIZE)

print("\n### Split 3 (val as a subset of training pool) ###")
print(f"Pool : {len(idx_pool)}\nTrain : {len(split3_train)}"
      f"\nVal : {len(split3_val)}\nTest : {len(split3_test)}")
print(f"\nVal subset Pool: {set(idx_val3).issubset(set(idx_pool))}")


# 4. Decode predictions back to human-readable labels
def decode(encoded_label: int) -> dict:
    full   = le.inverse_transform([encoded_label])[0]
    parts  = full.split("___")
    return {"species": parts[0], "disease": parts[1], "full": full}

print("\nLabel decode example:", decode(0))

### CHARTS GENERATION

COLORS = {"Train": "blue", "Val": "yellow", "Test": "green"}
CLASS_NAMES = le.classes_
N_CLASSES   = len(CLASS_NAMES)

def class_counts(indices):
    """Return count of each class for a given set of indices."""
    labels = all_labels_enc[indices]
    return np.bincount(labels, minlength=N_CLASSES)

# ── data for charts ───────────────────────────────────────────────────────────
counts = {
    "Split 1": {
        "Train": class_counts(idx_train),
        "Val":   class_counts(idx_val),
        "Test":  class_counts(idx_test),
    },
    "Split 2": {                          # same indices, aug only affects transforms
        "Train": class_counts(idx_train),
        "Val":   class_counts(idx_val),
        "Test":  class_counts(idx_test),
    },
    "Split 3": {
        "Train": class_counts(idx_train3),
        "Val":   class_counts(idx_val3),
        "Test":  class_counts(idx_test3),
    },
}

totals = {
    split: {role: c.sum() for role, c in roles.items()}
    for split, roles in counts.items()
}

# CHART 1 — Split size comparison
fig, ax = plt.subplots(figsize=(8, 4))
splits = list(totals.keys())
roles = ["Train", "Val", "Test"]
x = np.arange(len(splits))
bar_w = 0.22

for i, role in enumerate(roles):
    vals = [totals[s][role] for s in splits]
    bars = ax.bar(x + i * bar_w, vals, bar_w, label=role,
                  color=COLORS[role], zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                str(v), ha="center", va="bottom", fontsize=8)

ax.set_xticks(x + bar_w)
ax.set_xticklabels(splits)
ax.set_ylabel("Number of images")
ax.set_title("Chart 1 — Image count per set across all splits")
ax.legend()
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(PLOT_DIR/"chart1_split_sizes.png", dpi=150)
plt.show()

# CHART 2 — Proportions as stacked bar (one row per split)
fig, ax = plt.subplots(figsize=(8, 2.5))
total_imgs = len(all_paths)

for i, split in enumerate(splits):
    left = 0
    for role in roles:
        val = totals[split][role]
        pct = val / total_imgs * 100
        ax.barh(i, pct, left=left, color=COLORS[role], height=0.5)
        if pct > 3:
            ax.text(left + pct / 2, i, f"{role}\n{val} ({pct:.0f}%)",
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        left += pct

ax.set_yticks(range(len(splits)))
ax.set_yticklabels(splits)
ax.set_xlabel("% of total dataset")
ax.set_title("Chart 2 — Split proportions")
ax.set_xlim(0, 100)
patches = [mpatches.Patch(color=COLORS[r], label=r) for r in roles]
ax.legend(handles=patches, loc="lower right")
plt.tight_layout()
plt.savefig(PLOT_DIR/"chart2_proportions.png", dpi=150)
plt.show()

# CHART 3 — Per-class distribution in Train / Val / Test (Split 1 & 3) Shows whether stratification worked correctly
fig, axes = plt.subplots(1, 2, figsize=(14, max(4, N_CLASSES * 0.35 + 1)), sharey=True)
short_names = [c.replace("___", "\n") for c in CLASS_NAMES]
y = np.arange(N_CLASSES)
bar_w = 0.28

for ax, split_name, split_counts in zip(
    axes,
    ["Split 1 (separate val)", "Split 3 (val in pool)"],
    [counts["Split 1"], counts["Split 3"]],
):
    for i, role in enumerate(roles):
        ax.barh(y + (i - 1) * bar_w, split_counts[role], bar_w,
                label=role, color=COLORS[role], alpha=0.9)
    ax.set_title(f"Chart 3 — Class distribution\n{split_name}")
    ax.set_xlabel("Images per class")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

axes[0].set_yticks(y)
axes[0].set_yticklabels(short_names, fontsize=7)
plt.tight_layout()
plt.savefig(PLOT_DIR/"chart3_class_distribution.png", dpi=150)
plt.show()

# CHART 4 — Class balance ratio (max class / min class per set) A ratio of 1.0 = perfectly balanced. Higher = more imbalanced.
fig, ax = plt.subplots(figsize=(8, 4))
split_labels, role_labels, ratios = [], [], []

for split in splits:
    for role in roles:
        c = counts[split][role]
        ratio = c.max() / max(c.min(), 1)
        split_labels.append(split)
        role_labels.append(role)
        ratios.append(round(ratio, 2))

x = np.arange(len(splits))
bar_w = 0.22

for i, role in enumerate(roles):
    idxs = [j for j, r in enumerate(role_labels) if r == role]
    vals = [ratios[j] for j in idxs]
    bars = ax.bar(x + i * bar_w, vals, bar_w, label=role, color=COLORS[role], zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)

ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Perfect balance (1.0)")
ax.set_xticks(x + bar_w)
ax.set_xticklabels(splits)
ax.set_ylabel("Max class count / Min class count")
ax.set_title("Chart 4 — Class imbalance ratio per set\n(closer to 1.0 = better stratification)")
ax.legend()
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(PLOT_DIR/"chart4_balance_ratio.png", dpi=150)
plt.show()

print("\nCharts saved: chart1_split_sizes.png, chart2_proportions.png, "
      "chart3_class_distribution.png, chart4_balance_ratio.png")
