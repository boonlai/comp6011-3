import sys
import os
import shutil
import tempfile
import torch
import roboflow

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from util.finetune import get_finetuned_model
from util.preprocessing import extract_window, plot_and_save_ecg_window, load_image
from util.grad_cam import grad_cam

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path>")
        sys.exit(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(file_path: str) -> tuple[str, float]:
    """
    Given the path to an ECG record (.dat/.npy), predict the class of the record and output the Grad-CAM image.
    """
    # Remove extension from .dat if present
    if file_path.endswith(".dat"):
        file_path = os.path.splitext(file_path)[0]

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    Xi = extract_window(file_path).flatten()

    # Reshape back into windows
    win = Xi.reshape(200, 12)
    image_data = None

    # Save as image first
    with tempfile.TemporaryDirectory() as temp_dir:
        out_path = os.path.join(temp_dir, f"{base_name}.png")
        plot_and_save_ecg_window(win, out_path)
        image_data = load_image(out_path).to(device)

    # Prediction
    classes = ["1AVB", "AFIB", "AFLT", "LBBB", "RBBB", "NORM", "OTHERS"]
    model = get_finetuned_model(device, len(classes), silent=True)

    with torch.no_grad():
        logits = model(image_data)
        pred_idx = logits.squeeze(0).argmax().item()

    idx_to_class = {i: cls for i, cls in enumerate(sorted(classes))}
    confidence = torch.softmax(logits.squeeze(0), dim=0).max().item()
    pred_cls = idx_to_class[pred_idx]

    # Generate Grad-CAM image
    grad_cam(
        model,
        input_tensor=image_data,
        target_class=pred_idx,
        output_path=f"grad_cam/{base_name}.png",
        target_name=pred_cls,
        silent=True,
    )

    # Return the predicted class and its confidence
    return pred_cls, confidence


def predict_walk(path: str):
    """
    Given a path to a directory or file, predict the class of the record.
    """
    file_paths = []

    if os.path.isfile(path):
        if path.endswith(".npy"):
            file_paths.append(path)
    else:
        # Recursively find all files ending with _lr or .npy
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".dat") or file.endswith(".npy"):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)

    if len(file_paths) == 0:
        print("[Error] No files found in folder. Files must end with `.dat` or `.npy`.")
        return

    # Delete grad_cam/
    if os.path.exists("grad_cam"):
        shutil.rmtree("grad_cam")

    # List of predictions (list of tuples of (file_path, predicted_label))
    predictions = []

    enumerable = (
        file_paths if len(file_paths) == 1 else tqdm(file_paths, desc="Predicting")
    )

    for file_path in enumerable:
        predictions.append((file_path, predict(file_path)))

    # Output predictions
    print("\n" + "=" * 60)
    print("ECG PREDICTION RESULTS".center(60))
    print("=" * 60)

    for file_path, (predicted_class, confidence) in predictions:
        filename = os.path.basename(file_path)
        print(f"{filename:<30} -> {predicted_class} ({confidence:.2%})")

    print("=" * 60)
    print(f"Total files processed: {len(predictions)}")


predict_walk(sys.argv[1])
