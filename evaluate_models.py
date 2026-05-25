import json
from pathlib import Path

import numpy as np
import torch
import timm
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from dataset_preparation import get_label_encoder, get_split1_loaders, get_split2_loaders, get_split3_loaders

RESULTS_DIR = Path("training_results")
MODELS_DIR = Path("models")
RESULTS_DIR.mkdir(exist_ok=True)


def load_checkpoint(path: Path, num_classes: int, device: torch.device):
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_model(name: str, loader_fn, ck_path: Path, device: torch.device):
    print(f"Evaluating {name} using {ck_path}")
    le = get_label_encoder()
    num_classes = len(le.classes_)

    model = load_checkpoint(ck_path, num_classes=num_classes, device=device)

    # Build test loader
    _, loader_val, _ = loader_fn(batch_size=64, num_workers=0, img_size=224)

    ys = []
    ypred = []

    with torch.no_grad():
        for images, labels in tqdm(loader_val, desc=f"{name} test"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            ys.extend(labels.numpy().tolist())
            ypred.extend(preds.tolist())

    ys = np.array(ys, dtype=int)
    ypred = np.array(ypred, dtype=int)

    cm = confusion_matrix(ys, ypred, labels=range(num_classes))
    report = classification_report(ys, ypred, target_names=list(get_label_encoder().classes_), output_dict=True)

    # Save outputs
    np.save(RESULTS_DIR / f"confusion_{name}.npy", cm)
    with open(RESULTS_DIR / f"report_{name}.json", "w") as f:
        json.dump(report, f, indent=2)

    # Also save a small summary
    summary = {
        "accuracy": float((ys == ypred).mean()),
        "samples": int(len(ys)),
    }
    with open(RESULTS_DIR / f"metrics_{name}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved confusion_{name}.npy, report_{name}.json, metrics_{name}.json")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tasks = [
        ("split1", get_split1_loaders, MODELS_DIR / "MODEL1.pth"),
        ("split2", get_split2_loaders, MODELS_DIR / "MODEL2.pth"),
        ("split3", get_split3_loaders, MODELS_DIR / "MODEL3.pth"),
        ("overfit", get_split1_loaders, MODELS_DIR / "MODEL_OVERFIT.pth"),
    ]

    for name, loader_fn, ck in tasks:
        if not ck.exists():
            print(f"  [SKIP] checkpoint not found: {ck}")
            continue
        try:
            evaluate_model(name, loader_fn, ck, device)
        except Exception as e:
            print(f"  [ERROR] evaluating {name}: {e}")


if __name__ == "__main__":
    main()
