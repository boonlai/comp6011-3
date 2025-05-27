import os
import shutil

import roboflow
from sklearn.metrics import accuracy_score
import torch
import torchvision.transforms as T
from tqdm import tqdm

from finetune import get_finetuned_model
from util.preprocessing import load_image

# Must be ran from `python train.py`
if __name__ != "__main__":
    exit()


def download_from_roboflow() -> None:
    rf_workspace = "personal-g6wmi"
    rf_project = "research-kii6w"
    rf_version = 5

    roboflow.login()

    rf = roboflow.Roboflow()

    project = rf.workspace(rf_workspace).project(rf_project)
    project.version(rf_version).download("folder")

    # Move Research folder into data directory
    shutil.move(f"Research-{rf_version}/", f"data/")


def get_files_to_label() -> dict[str, str]:
    if not os.path.exists("data/"):
        download_from_roboflow()

    cwd = os.getcwd()

    ROOT_DIR = os.path.join(cwd, f"data/train")

    labels = {}

    for folder in os.listdir(ROOT_DIR):
        for file in os.listdir(os.path.join(ROOT_DIR, folder)):
            if file.endswith(".jpg"):
                full_name = os.path.join(ROOT_DIR, folder, file)
                labels[full_name] = folder

    return labels


labels = get_files_to_label()
files = labels.keys()

# List of all classes
classes = ["1AVB", "AFIB", "AFLT", "LBBB", "RBBB", "NORM", "OTHERS"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_finetuned_model(device, len(classes), train=True)

# Gather all (true_label, filepath) pairs
all_samples = []
for cls in classes:
    folder = f"data/test/{cls}"
    if not os.path.isdir(folder):
        print(f"No folder found for class {cls} at {folder}")
        continue
    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".png")):
            all_samples.append((cls, os.path.join(folder, fname)))

if not all_samples:
    raise RuntimeError("No image files found in any class folders!")

y_true, y_pred = [], []

# Iterate with progress bar
for true_cls, img_path in tqdm(all_samples, desc="Processing images"):
    # load_image should perform the same transforms as during training
    new_image = load_image(img_path).to(device)
    with torch.no_grad():
        logits = model(new_image)
        pred_idx = logits.squeeze(0).argmax().item()

    idx_to_class = {i: cls for i, cls in enumerate(sorted(classes))}
    pred_cls = idx_to_class[pred_idx]

    y_true.append(true_cls)
    y_pred.append(pred_cls)

# Compute and print per-class accuracy
print("\nPer-class accuracy:")
for cls in classes:
    idxs = [i for i, t in enumerate(y_true) if t == cls]
    if not idxs:
        print(f"  {cls}: no samples")
        continue
    acc = sum(1 for i in idxs if y_pred[i] == cls) / len(idxs)
    print(f"  {cls}: {acc:.2%}")

# Overall accuracy
overall = accuracy_score(y_true, y_pred)
print(f"\nOverall accuracy: {overall:.2%}")
