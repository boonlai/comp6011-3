# Following: https://ieeexplore.ieee.org/document/10590559

import numpy as np
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from scipy.signal import butter, sosfiltfilt, resample


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


def extract_window(
    record_path: str, fs_target: float = 100.0, window_sec: int = 2
) -> np.ndarray:
    if record_path.endswith(".npy"):
        # It's in shape of (12, n_samples)
        # So we should transpose it
        sig = np.load(record_path).T
        meta = {"fs": fs_target}
    else:
        sig, meta = wfdb.rdsamp(record_path)
    sig_pp = bandpass_and_resample(sig, meta.get("fs", fs_target), fs_target)
    start = int(window_sec * fs_target)
    end = start + int(window_sec * fs_target)
    return sig_pp[start:end, :]


def plot_and_save_ecg_window(
    sig_win: np.ndarray,
    output_path: str,
    fs: float = 100.0,
    resize_px: int = 518,
    dpi: int = 150,
):
    times = np.arange(sig_win.shape[0]) / fs
    fig, axes = plt.subplots(4, 3, figsize=(6, 6))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < sig_win.shape[1]:
            ax.plot(times, sig_win[:, i], lw=0.5, color="black")
        ax.axis("off")
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    tmp = output_path + ".tmp.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(tmp, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    img = Image.open(tmp).convert("L")
    img = img.resize((resize_px, resize_px), Image.BILINEAR)
    img.save(output_path)
    os.remove(tmp)
