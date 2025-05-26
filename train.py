import glob
import json
import os
import shutil

import numpy as np
from PIL import Image
import roboflow
from sklearn import svm
from sklearn.metrics import accuracy_score
import torch
import torchvision.transforms as T
from tqdm import tqdm


def load_image(img: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    # img = Image.open(img)
    # Since we saved image in "L" mode, maybe try converting to RGB
    img = Image.open(img).convert("RGB")
    transformed_img = transform_image(img)[:3].unsqueeze(0)
    return transformed_img


def compute_embeddings(files: list) -> dict:
    """
    Create an index that contains all of the images in the specified list of files.
    """
    all_embeddings = {}

    with torch.no_grad():
        for i, file in enumerate(tqdm(files)):
            embeddings = dinov2_vits14(load_image(file).to(device))
            all_embeddings[file] = np.array(
                embeddings[0].cpu().numpy()).reshape(1, -1).tolist()

    with open("all_embeddings.json", "w") as f:
        f.write(json.dumps(all_embeddings))

    return all_embeddings


rf_workspace = "personal-g6wmi"
rf_project = "research-kii6w"
rf_version = 5

roboflow.login()

shutil.rmtree(f"Research-{rf_version}/", ignore_errors=True)

rf = roboflow.Roboflow()

project = rf.workspace(rf_workspace).project(rf_project)
dataset = project.version(rf_version).download("folder")

cwd = os.getcwd()

ROOT_DIR = os.path.join(cwd, f"Research-{rf_version}/train")

labels = {}

for folder in os.listdir(ROOT_DIR):
    for file in os.listdir(os.path.join(ROOT_DIR, folder)):
        if file.endswith(".jpg"):
            full_name = os.path.join(ROOT_DIR, folder, file)
            labels[full_name] = folder

files = labels.keys()

dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

dinov2_vits14.to(device)
dinov2_vits14.eval()  # Optimization

transform_image = T.Compose([
    T.ToTensor(),
    T.Resize(518),
    T.CenterCrop(518),
    # T.Normalize([0.5], [0.5])
    # Normalize across three channels now
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])  # Native is 518x518

with torch.no_grad():  # Not required to compute embeddings
    embeddings = compute_embeddings(files)

# clf = svm.SVC(gamma='scale')
clf = svm.SVC(kernel='linear', C=1.0)  # Try a different kernel

y = [labels[file] for file in files]

embedding_list = list(embeddings.values())

clf.fit(np.array(embedding_list).reshape(-1, 384), y)

# List of all classes
classes = ["1AVB", "AFIB", "AFLT", "LBBB", "RBBB", "NORM", "OTHERS"]

# Gather all (true_label, filepath) pairs
all_samples = []
for cls in classes:
    folder = f"Research-{rf_version}/test/{cls}"
    if not os.path.isdir(folder):
        print(f"No folder found for class {cls} at {folder}")
        continue
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.png')):
            all_samples.append((cls, os.path.join(folder, fname)))

if not all_samples:
    raise RuntimeError("No image files found in any class folders!")

y_true, y_pred = [], []

# Iterate with progress bar
for true_cls, img_path in tqdm(all_samples, desc="Processing images"):
    # load_image should perform the same transforms as during training
    new_image = load_image(img_path)
    with torch.no_grad():
        embedding = dinov2_vits14(new_image.to(device))
        pred_cls = clf.predict(np.array(embedding[0].cpu()).reshape(1, -1))[0]

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
