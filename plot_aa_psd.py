import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch, iirnotch, filtfilt

DEFAULT_FILE = r"C:\Users\sereg\Downloads\faces_basic\faces_basic\data\aa\aa_faceshouses.mat"
NOTCH_FREQS = [60, 120, 180, 240]
FMIN, FMAX = 1, 250


def subject_from_path(file_path):
    path = Path(file_path)
    name = path.stem
    return name.replace("_faceshouses", "")


parser = argparse.ArgumentParser(description="Plot frequency spectrum before/after notch filtering.")
parser.add_argument("file_path", nargs="?", default=DEFAULT_FILE, help="Path to *_faceshouses.mat")
args = parser.parse_args()

file_path = args.file_path
subject = subject_from_path(file_path)

mat = loadmat(file_path, squeeze_me=True)
data = np.asarray(mat["data"], dtype=float)  # time x channels
srate = float(mat["srate"])

freqs, psd_before = welch(data, fs=srate, axis=0, nperseg=4096)
asd_before = np.sqrt(psd_before)

filtered = data.copy()
used_freqs = []
for f0 in NOTCH_FREQS:
    if f0 < srate / 2:
        b, a = iirnotch(w0=f0, Q=30, fs=srate)
        filtered = filtfilt(b, a, filtered, axis=0)
        used_freqs.append(f0)

freqs_after, psd_after = welch(filtered, fs=srate, axis=0, nperseg=4096)
asd_after = np.sqrt(psd_after)

print("Subject:", subject)
print("File:", file_path)
print("Sampling rate:", srate, "Hz")
print("Notch frequencies:", used_freqs)
print("\nPeak check:")
for f0 in used_freqs:
    i1 = np.argmin(np.abs(freqs - f0))
    i2 = np.argmin(np.abs(freqs_after - f0))
    before = np.max(asd_before[i1, :])
    after = np.max(asd_after[i2, :])
    print(f"{f0:>3} Hz: before={before:.6g}, after={after:.6g}, ratio={after / before:.6g}")

mask = (freqs >= FMIN) & (freqs <= FMAX)
mask_after = (freqs_after >= FMIN) & (freqs_after <= FMAX)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(freqs[mask], asd_before[mask, :], color="black", alpha=0.30, linewidth=0.7)
axes[0].set_title("BEFORE notch filter")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True, alpha=0.35)

axes[1].plot(freqs_after[mask_after], asd_after[mask_after, :], color="black", alpha=0.30, linewidth=0.7)
axes[1].set_title("AFTER notch filter: 60, 120, 180, 240 Hz")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True, alpha=0.35)

for ax in axes:
    for f0 in used_freqs:
        ax.axvline(f0, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
        ymax_for_label = ax.get_ylim()[1]
        ax.text(f0, ymax_for_label * 0.95, f"{f0} Hz", color="red", ha="center", va="top", fontsize=8)

ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
axes[0].set_ylim(0, ymax)
axes[1].set_ylim(0, ymax)

fig.suptitle(f"Subject {subject}: frequency spectrum before/after notch", y=0.98)
fig.tight_layout()
plt.show()
