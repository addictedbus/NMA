"""
Run from VS Code terminal.

Install dependencies once:
    pip install numpy scipy

Usage:
    python save_notched_mat.py "<path-to-subject>\aa_faceshouses.mat"

This creates a new file next to the original:
    aa_faceshouses_notch60.mat

The original .mat file is not changed.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import iirnotch, filtfilt

NOTCH_FREQS = [60, 120, 180, 240]


def make_output_path(file_path):
    path = Path(file_path)
    return path.with_name(path.stem + "_notch60" + path.suffix)


def main():
    parser = argparse.ArgumentParser(description="Save a notch-filtered copy of *_faceshouses.mat.")
    parser.add_argument("file_path", help="Path to *_faceshouses.mat")
    parser.add_argument("--output", default=None, help="Optional output .mat path")
    args = parser.parse_args()

    file_path = Path(args.file_path)
    output_path = Path(args.output) if args.output else make_output_path(file_path)

    mat = loadmat(file_path, squeeze_me=True)
    data = np.asarray(mat["data"], dtype=float)  # time x channels
    stim = np.asarray(mat["stim"])
    srate = float(mat["srate"])

    filtered = data.copy()
    used_freqs = []
    for f0 in NOTCH_FREQS:
        if f0 < srate / 2:
            b, a = iirnotch(w0=f0, Q=30, fs=srate)
            filtered = filtfilt(b, a, filtered, axis=0)
            used_freqs.append(f0)

    out = {
        "data": filtered,
        "stim": stim,
        "srate": srate,
        "notch_freqs": np.asarray(used_freqs, dtype=float),
        "source_file": str(file_path),
    }

    savemat(output_path, out)

    print("Source:", file_path)
    print("Saved:", output_path)
    print("Data shape:", filtered.shape, "(time x channels)")
    print("Sampling rate:", srate, "Hz")
    print("Applied notch frequencies:", used_freqs)


if __name__ == "__main__":
    main()
