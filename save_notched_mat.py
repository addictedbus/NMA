r"""Create a paper-like line-noise-filtered copy of a faces-houses recording.

Run from a VS Code terminal:
    python save_notched_mat.py "<path-to-subject>\ja_faceshouses.mat"

The original file is not changed. The output is saved beside it as:
    ja_faceshouses_notch58_62.mat

Miller et al. described a third-order Butterworth notch filter from 58 to
62 Hz. This script applies that band-stop filter in both directions with
sosfiltfilt to avoid phase shifts in the offline signal.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import butter, sosfiltfilt


NOTCH_BAND = (58.0, 62.0)
FILTER_ORDER = 3


def make_output_path(file_path):
    path = Path(file_path)
    return path.with_name(path.stem + "_notch58_62" + path.suffix)


def main():
    parser = argparse.ArgumentParser(
        description="Save a paper-like 58-62 Hz band-stop copy of *_faceshouses.mat."
    )
    parser.add_argument("file_path", help="Path to the original *_faceshouses.mat")
    parser.add_argument("--output", help="Optional output .mat path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    args = parser.parse_args()

    file_path = Path(args.file_path).resolve()
    if "_notch" in file_path.stem.lower():
        raise ValueError(
            "The input filename already appears to be notch-filtered. Use the original "
            "*_faceshouses.mat file to avoid applying the filter twice."
        )
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    output_path = Path(args.output).resolve() if args.output else make_output_path(file_path)
    if output_path.exists() and not args.overwrite:
        print("Notch-filtered file already exists:", output_path)
        print("Nothing was changed. Use this file with inspect_raw.py.")
        return

    mat = loadmat(file_path, squeeze_me=True)
    data = np.asarray(mat["data"], dtype=float)
    stim = np.asarray(mat["stim"])
    srate = float(mat["srate"])

    if NOTCH_BAND[1] >= srate / 2:
        raise ValueError(
            f"The 58-62 Hz band is invalid for sampling rate {srate:g} Hz."
        )

    sos = butter(
        N=FILTER_ORDER,
        Wn=NOTCH_BAND,
        btype="bandstop",
        fs=srate,
        output="sos",
    )
    filtered = sosfiltfilt(sos, data, axis=0)

    savemat(
        output_path,
        {
            "data": filtered,
            "stim": stim,
            "srate": srate,
            "notch_band": np.asarray(NOTCH_BAND, dtype=float),
            "filter_order": FILTER_ORDER,
            "filter_family": "Butterworth",
            "filter_mode": "zero-phase sosfiltfilt",
            "source_file": str(file_path),
        },
    )

    print("Source:", file_path)
    print("Saved:", output_path)
    print("Data shape:", filtered.shape, "(time x channels)")
    print("Sampling rate:", srate, "Hz")
    print("Filter: third-order Butterworth band-stop")
    print("Band:", list(NOTCH_BAND), "Hz")
    print("Application: zero-phase sosfiltfilt")


if __name__ == "__main__":
    main()
