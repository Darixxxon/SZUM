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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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


def collect_samples(root: Path):
    paths, labels = [], []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        seen = set()
        for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
            for img_path in class_dir.glob(ext):
                resolved = str(img_path).lower() 
                if resolved not in seen:
                    seen.add(resolved)
                    paths.append(img_path)
                    labels.append(class_dir.name)
    return np.array(paths), np.array(labels)


_cache = {}


def _prepare_data():
    if _cache:
        return _cache

    all_paths, all_labels = collect_samples(DATASET_ROOT)

    le = LabelEncoder()
    all_labels_enc = le.fit_transform(all_labels)

    species = sorted({l.split("___")[0] for l in le.classes_})
    diseases = sorted({l.split("___")[1] for l in le.classes_})
    print(f"Classes  : {len(le.classes_)}")
    print(f"Species  : {species}")
    print(f"Diseases : {diseases}")
    print(f"Total    : {len(all_paths)} images")

    idx = np.arange(len(all_paths))

    idx_train, idx_temp = train_test_split(
        idx, test_size=0.2, stratify=all_labels_enc, random_state=RANDOM_STATE
    )
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=0.5,
        stratify=all_labels_enc[idx_temp],
        random_state=RANDOM_STATE,
    )

    print(
        f"\nBase split: \nTrain: {len(idx_train)}\nVal:{len(idx_val)}\nTest:{len(idx_test)}"
    )

    idx_val3, _ = train_test_split(
        idx_train,
        test_size=8 / 9,
        stratify=all_labels_enc[idx_train],
        random_state=RANDOM_STATE,
    )

    if SAVE_SPLITS:
        print("Saving splits...")
        np.savez(
            SPLIT_FILE,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            idx_val3=idx_val3,
        )

    _cache.update(
        dict(
            all_paths=all_paths,
            all_labels=all_labels,
            all_labels_enc=all_labels_enc,
            le=le,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            idx_val3=idx_val3,
        )
    )
    return _cache


def get_raw_transform(img_size=IMG_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )


def get_train_transform(img_size=IMG_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
            ),
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_transform(img_size=IMG_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

def get_num_classes():
    data = _prepare_data()
    return len(data["le"].classes_)


def get_label_encoder():
    data = _prepare_data()
    return data["le"]


def get_split1_loaders(batch_size=BATCH_SIZE, fraction=1.0, num_workers=0, img_size=IMG_SIZE):
    data = _prepare_data()
    raw_tf = get_raw_transform(img_size)

    idx_train = data["idx_train"]
    if fraction < 1.0:
        idx_sub, _ = train_test_split(
            idx_train,
            train_size=fraction,
            stratify=data["all_labels_enc"][idx_train],
            random_state=RANDOM_STATE,
        )
        idx_train = idx_sub

    full_ds = LeafDataset(data["all_paths"], data["all_labels_enc"], transform=raw_tf)
    train_ds = Subset(full_ds, idx_train)
    val_ds = Subset(full_ds, data["idx_val"])
    test_ds = Subset(full_ds, data["idx_test"])

    loader_train = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    loader_val = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    loader_test = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    print(f"\n### Split 1 (raw, fraction={fraction}, img_size={img_size}) ###")
    print(f"Train : {len(train_ds)}\nVal : {len(val_ds)}\nTest : {len(test_ds)}")

    return loader_train, loader_val, loader_test


def get_split2_loaders(batch_size=BATCH_SIZE, num_workers=4, img_size=IMG_SIZE):
    data = _prepare_data()
    train_tf = get_train_transform(img_size)
    eval_tf = get_eval_transform(img_size)

    train_ds = LeafDataset(
        data["all_paths"][data["idx_train"]],
        data["all_labels_enc"][data["idx_train"]],
        train_tf,
    )
    val_ds = LeafDataset(
        data["all_paths"][data["idx_val"]],
        data["all_labels_enc"][data["idx_val"]],
        eval_tf,
    )
    test_ds = LeafDataset(
        data["all_paths"][data["idx_test"]],
        data["all_labels_enc"][data["idx_test"]],
        eval_tf,
    )

    loader_train = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    loader_val = DataLoader(
        val_ds, batch_size=batch_size, num_workers=num_workers,
    )
    loader_test = DataLoader(
        test_ds, batch_size=batch_size, num_workers=num_workers,
    )

    print(f"\n### Split 2 (normalised + augmented, img_size={img_size}) ###")
    print(f"Train : {len(train_ds)}\nVal : {len(val_ds)}\nTest : {len(test_ds)}")

    return loader_train, loader_val, loader_test


def get_split3_loaders(batch_size=BATCH_SIZE, num_workers=4, img_size=IMG_SIZE):
    data = _prepare_data()
    train_tf = get_train_transform(img_size)
    eval_tf = get_eval_transform(img_size)

    train_ds = LeafDataset(
        data["all_paths"][data["idx_train"]],
        data["all_labels_enc"][data["idx_train"]],
        train_tf,
    )
    val_ds = LeafDataset(
        data["all_paths"][data["idx_val3"]],
        data["all_labels_enc"][data["idx_val3"]],
        eval_tf,
    )
    test_ds = LeafDataset(
        data["all_paths"][data["idx_test"]],
        data["all_labels_enc"][data["idx_test"]],
        eval_tf,
    )

    loader_train = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    loader_val = DataLoader(
        val_ds, batch_size=batch_size, num_workers=num_workers,
    )
    loader_test = DataLoader(
        test_ds, batch_size=batch_size, num_workers=num_workers,
    )

    print(f"\n### Split 3 (val as a subset of training pool, img_size={img_size}) ###")
    print(
        f"Train : {len(train_ds)}"
        f"\nVal : {len(val_ds)}"
        f"\nTest : {len(test_ds)}"
    )
    print(
        f"\nVal subset Train: {set(data['idx_val3']).issubset(set(data['idx_train']))}"
    )

    return loader_train, loader_val, loader_test


def decode(encoded_label: int) -> dict:
    data = _prepare_data()
    le = data["le"]
    full = le.inverse_transform([encoded_label])[0]
    parts = full.split("___")
    return {"species": parts[0], "disease": parts[1], "full": full}


if __name__ == "__main__":
    data = _prepare_data()

    l1_train, l1_val, l1_test = get_split1_loaders()
    l2_train, l2_val, l2_test = get_split2_loaders()
    l3_train, l3_val, l3_test = get_split3_loaders()

    print(f"\nNum classes: {get_num_classes()}")
    print("Label decode example:", decode(0))
