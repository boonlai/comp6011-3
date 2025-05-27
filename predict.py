import os
import tempfile
import torch
from train import get_finetuned_model
from util.preprocessing import extract_window, plot_and_save_ecg_window, load_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(file_path: str) -> str:
    """
    Given the path to an ECG record (.dat/.npy), predict the class of the record.
    """
    # Remove extension from .dat if present
    if file_path.endswith(".dat"):
        file_path = os.path.splitext(file_path)[0]

    Xi = extract_window(file_path).flatten()

    # Reshape back into windows
    win = Xi.reshape(200, 12)
    image_data = None

    # Save as image first
    with tempfile.TemporaryDirectory() as temp_dir:
        out_path = os.path.join(
            temp_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}.png"
        )
        plot_and_save_ecg_window(win, out_path)
        image_data = load_image(out_path).to(device)

    # Prediction
    classes = ["1AVB", "AFIB", "AFLT", "LBBB", "RBBB", "NORM", "OTHERS"]
    model = get_finetuned_model(device, len(classes))

    with torch.no_grad():
        logits = model(image_data)
        pred_idx = logits.squeeze(0).argmax().item()

    idx_to_class = {i: cls for i, cls in enumerate(sorted(classes))}
    pred_cls = idx_to_class[pred_idx]

    # Return the predicted class
    return pred_cls


def predict_walk(path: str):
    """ """
    # List of predictions (list of tuples of (file_path, predicted_label))
    predictions = []

    if os.path.isfile(path):
        if path.endswith(".npy"):
            predictions.append((path, predict(path)))
    else:
        # Recursively find all files ending with _lr or .npy
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".dat") or file.endswith(".npy"):
                    file_path = os.path.join(root, file)
                    print(f"Extracting from {file_path}")
                    predictions.append((file_path, predict(file_path)))

    if len(predictions) == 0:
        print("[Error] No files found in folder. Files must end with `.dat` or `.npy`.")
        return

    # Output predictions
    print("\n" + "=" * 60)
    print("ECG PREDICTION RESULTS")
    print("=" * 60)

    for file_path, predicted_class in predictions:
        filename = os.path.basename(file_path)
        print(f"{filename:<30} -> {predicted_class}")

    print("=" * 60)
    print(f"Total files processed: {len(predictions)}")


if __name__ == "__main__":
    """
    Extract windows from ECG records in a directory or single file, and output their predictions.

    Args:
        path: Path to directory containing records or direct path to .npy file

    Returns:
        np.ndarray: Stacked windows array
    """
    import sys

    if len(sys.argv) != 2:
        print("Usage: python predict.py <path>")
        sys.exit(1)

    predict_walk(sys.argv[1])
