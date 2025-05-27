import os
from tqdm import tqdm
from util.preprocessing import extract_window, plot_and_save_ecg_window
import numpy as np


def predict(file_path: str) -> str:
    # Remove extension if present
    if file_path.endswith(".dat"):
        file_path = os.path.splitext(file_path)[0]
    Xi = extract_window(file_path).flatten()

    # Reshape back into windows
    win = Xi.reshape(200, 12)

    out_path = f"predict_imgs/{os.path.splitext(os.path.basename(file_path))[0]}.png"
    plot_and_save_ecg_window(win, out_path)

    return "[Prediction here]"


def predict_walk(path: str):
    """
    Extract windows from ECG records in a directory or single file.

    Args:
        path: Path to directory containing records or direct path to .npy file

    Returns:
        np.ndarray: Stacked windows array
    """
    os.makedirs("predict_imgs/", exist_ok=True)

    X_windows = []

    if os.path.isfile(path):
        if path.endswith(".npy"):
            X_windows.append(predict(path))
    else:
        # Recursively find all files ending with _lr or .npy
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".dat") or file.endswith(".npy"):
                    file_path = os.path.join(root, file)
                    print(f"Extracting from {file_path}")
                    X_windows.append(predict(file_path))

    if len(X_windows) == 0:
        print("[Error] No files found in folder. Files must end with `.dat` or `.npy`.")
        return

    X = np.stack(X_windows, axis=0)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python predict.py <path>")
        sys.exit(1)
    predict_walk(sys.argv[1])
