import wfdb
import os
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample
import matplotlib.pyplot as plt
from PIL import Image

# LR => 100hz
# HR => 500hz
# Based on the methodology from: https://ieeexplore.ieee.org/document/10590559


def bandpass_and_resample(
    sig: np.ndarray,
    fs_orig: float,
    fs_target: float = 100.0,
    lowcut: float = 0.5,
    highcut: float = 40.0,
) -> np.ndarray:
    """
    3rd-order Butterworth bandpass at [lowcut, highcut] Hz,
    then resample to fs_target if fs_orig != fs_target.
    """
    # Design filter
    nyq = fs_orig / 2
    sos = butter(3, [lowcut / nyq, highcut / nyq], btype="band", output="sos")

    # Apply forward-backward filtering
    filtered = sosfiltfilt(sos, sig, axis=0)

    # Resample if needed
    if fs_orig != fs_target:
        n_samples = int(filtered.shape[0] * fs_target / fs_orig)
        filtered = resample(filtered, n_samples, axis=0)

    return filtered


def plot_and_save_ecg_image(
    record_path: str,
    output_path: str,
    fs: float = 100.0,
    window_sec: int = 2,
    resize_px: int = 518,
    dpi: int = 150,
):
    """
    1) Load a low-res (100 Hz) PTB-XL record
    2) Preprocess (bandpass + resample)
    3) Take samples [2s:4s] → 2 s of clean data
    4) Plot 12 leads in 4×3, remove all axes/labels
    5) Save temp PNG, resize to (resize_px, resize_px), overwrite output
    """
    # Pull in the record
    sig, meta = wfdb.rdsamp(record_path)
    fs_orig = meta.get("fs", fs)

    # Preprocess the signal
    sig_pp = bandpass_and_resample(sig, fs_orig, fs)

    # Windowing
    start = int(window_sec * fs)
    end = start + int(window_sec * fs)
    sig_win = sig_pp[start:end, :]  # shape (200, 12)

    # Plot the signals in a 4x3 grid
    times = np.arange(sig_win.shape[0]) / fs  # in seconds
    fig, axes = plt.subplots(4, 3, figsize=(6, 6))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < sig_win.shape[1]:
            ax.plot(times, sig_win[:, i], lw=0.5, color="black")
        ax.axis("off")
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)

    # Save and resize the image for DINOv2
    tmp = output_path + ".tmp.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(tmp, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    img = Image.open(tmp).convert("L")
    img = img.resize((resize_px, resize_px), Image.BILINEAR)
    img.save(output_path)
    os.remove(tmp)
