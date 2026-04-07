import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

PLOT_DIR = Path("Plots")
PLOT_DIR.mkdir(exist_ok=True)
DATASET_ROOT = Path("Dataset")
IMG_SIZE = 256
BATCH_SIZE = 32
RANDOM_STATE = 42  # Set value for RANDOM_STATE which is used in train_test_split ensures reproducible data split
SAVE_SPLITS = True

SPLIT_DIR = Path("splits")
SPLIT_DIR.mkdir(exist_ok=True)
SPLIT_FILE = SPLIT_DIR / "splits.npz"

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
species = sorted({l.split("___")[0] for l in le.classes_})
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
    idx_temp,
    test_size=0.5,
    stratify=all_labels_enc[idx_temp],
    random_state=RANDOM_STATE,
)

# Verify base split
print(
    f"\nBase split: \nTrain: {len(idx_train)}\nVal:{len(idx_val)}\nTest:{len(idx_test)}"
)


class LeafDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


# SPLIT 1 — Raw  (only resize + tensor conversion, no normalisation)
raw_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
)

split1_full = LeafDataset(all_paths, all_labels_enc, transform=raw_transform)
split1_train = Subset(split1_full, idx_train)
split1_val = Subset(split1_full, idx_val)
split1_test = Subset(split1_full, idx_test)

loader1_train = DataLoader(split1_train, batch_size=BATCH_SIZE, shuffle=True)
loader1_val = DataLoader(split1_val, batch_size=BATCH_SIZE)
loader1_test = DataLoader(split1_test, batch_size=BATCH_SIZE)

print("\n### Split 1 (raw) ###")
print(
    f"Train : {len(split1_train)}\nVal : {len(split1_val)}\nTest : {len(split1_test)}"
)

# SPLIT 2 — Normalised + augmented
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# TRAIN — heavy augmentation to improve generalisation
# - horizontal/vertical flipping
# - rotation
# - differentation in color distribution
# - zoomed/distanced photos
# - blurring
train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

# VAL / TEST — only resize + normalise (no augmentation, deterministic)
eval_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

split2_train = LeafDataset(
    all_paths[idx_train], all_labels_enc[idx_train], train_transform
)
split2_val = LeafDataset(all_paths[idx_val], all_labels_enc[idx_val], eval_transform)
split2_test = LeafDataset(all_paths[idx_test], all_labels_enc[idx_test], eval_transform)

loader2_train = DataLoader(
    split2_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
loader2_val = DataLoader(split2_val, batch_size=BATCH_SIZE, num_workers=4)
loader2_test = DataLoader(split2_test, batch_size=BATCH_SIZE, num_workers=4)

print("\n### Split 2 (normalised + augmented) ###")
print(
    f"Train : {len(split2_train)}\nVal : {len(split2_val)}\nTest : {len(split2_test)}"
)


# SPLIT 3 — VAL is a subset of the train pool
# VAL drawn from inside the training pool (same 10% absolute size)
idx_val3, _ = train_test_split(
    idx_train,
    test_size=8 / 9,  # keeps around 10% of total size as val
    stratify=all_labels_enc[idx_train],
    random_state=RANDOM_STATE,
)

split3_train = LeafDataset(
    all_paths[idx_train], all_labels_enc[idx_train], train_transform
)
split3_val = LeafDataset(all_paths[idx_val3], all_labels_enc[idx_val3], eval_transform)
split3_test = LeafDataset(all_paths[idx_test], all_labels_enc[idx_test], eval_transform)

loader3_train = DataLoader(split3_train, batch_size=BATCH_SIZE, shuffle=True)
loader3_val = DataLoader(split3_val, batch_size=BATCH_SIZE)
loader3_test = DataLoader(split3_test, batch_size=BATCH_SIZE)

print("\n### Split 3 (val as a subset of training pool) ###")
print(
    f"Train : {len(split3_train)}"
    f"\nVal : {len(split3_val)}"
    f"\nTest : {len(split3_test)}"
)
print(f"\nVal subset Train: {set(idx_val3).issubset(set(idx_train))}")


# 4. Decode predictions back to human-readable labels
def decode(encoded_label: int) -> dict:
    full = le.inverse_transform([encoded_label])[0]
    parts = full.split("___")
    return {"species": parts[0], "disease": parts[1], "full": full}

if SAVE_SPLITS:
        print("Saving splits...")
        np.savez(
            SPLIT_FILE,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            idx_val3=idx_val3,
        )

print("\nLabel decode example:", decode(0))
