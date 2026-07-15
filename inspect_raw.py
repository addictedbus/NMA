r"""Interactive raw ECoG viewer with anatomical labels and bad-channel saving.

Run from a VS Code terminal after creating the paper-like filtered file:
    python inspect_raw.py "<path-to-subject>\ja_faceshouses_notch58_62.mat"

The script automatically looks for:
    <faces_basic_root>\locs\ja_xslocs.mat

Channel E1 is raw data column 1 / Python data[:, 0] / ElectrodeIndex 1.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from scipy.io import loadmat


DEFAULT_DURATION = 8.0
DEFAULT_N_CHANNELS = 25

AREA_LABELS = {
    1: "Temporal pole",
    2: "Parahippocampal gyrus",
    3: "Inferior temporal gyrus",
    4: "Middle temporal gyrus",
    5: "Fusiform gyrus",
    6: "Lingual gyrus",
    7: "Inferior occipital gyrus",
    8: "Cuneus",
    9: "Post-ventral cingulate gyrus",
    10: "Middle occipital gyrus",
    11: "Occipital pole",
    12: "Precuneus",
    13: "Superior occipital gyrus",
    14: "Post-dorsal cingulate gyrus",
    20: "Non-included area",
}

ROI_BY_AREA = {
    "Parahippocampal gyrus": "Medial temporal",
    "Fusiform gyrus": "Ventral temporal visual",
    "Inferior temporal gyrus": "Ventral temporal visual",
    "Lingual gyrus": "Occipital visual",
    "Cuneus": "Occipital visual",
    "Inferior occipital gyrus": "Occipital visual",
    "Middle occipital gyrus": "Occipital visual",
    "Superior occipital gyrus": "Occipital visual",
    "Middle temporal gyrus": "Temporal association",
    "Temporal pole": "Temporal association",
}


def subject_from_path(file_path):
    return Path(file_path).stem.split("_faceshouses", 1)[0]


def default_loc_path(data_path, subject):
    data_path = Path(data_path).resolve()
    for parent in data_path.parents:
        candidate = parent / "locs" / f"{subject}_xslocs.mat"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find locs/{subject}_xslocs.mat above {data_path}. "
        "Pass it explicitly with --loc-file."
    )


def load_data_and_localisation(data_path, loc_path):
    raw_mat = loadmat(data_path, squeeze_me=True)
    data = np.asarray(raw_mat["data"], dtype=float).T
    stim = np.asarray(raw_mat["stim"])
    srate = float(raw_mat["srate"])

    loc_mat = loadmat(loc_path, squeeze_me=True)
    elcode = np.atleast_1d(loc_mat["elcode"]).astype(int)
    locs = np.atleast_2d(np.asarray(loc_mat["locs"], dtype=float))

    n_channels = data.shape[0]
    if len(elcode) != n_channels or locs.shape != (n_channels, 3):
        raise ValueError(
            "Raw/localisation mismatch: "
            f"data has {n_channels} channels, elcode has {len(elcode)}, "
            f"and locs has shape {locs.shape}."
        )
    if stim.shape[0] != data.shape[1]:
        raise ValueError("Stimulus vector length does not match raw time points.")

    notch_band = None
    if "notch_band" in raw_mat:
        notch_band = np.atleast_1d(raw_mat["notch_band"]).astype(float)
    filter_order = None
    if "filter_order" in raw_mat:
        filter_order = int(np.asarray(raw_mat["filter_order"]).squeeze())

    return data, stim, srate, elcode, locs, notch_band, filter_order


def estimate_display_scale(data, srate):
    """Estimate a robust trace spacing from the first 30 seconds."""
    preview_samples = min(data.shape[1], max(1, int(30 * srate)))
    preview = data[:, :preview_samples]
    low = np.nanpercentile(preview, 1, axis=1)
    high = np.nanpercentile(preview, 99, axis=1)
    channel_ranges = high - low
    valid_ranges = channel_ranges[np.isfinite(channel_ranges) & (channel_ranges > 0)]
    if not len(valid_ranges):
        return 1.0
    return float(np.median(valid_ranges) * 1.25)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive ECoG viewer with anatomy and ROI labels."
    )
    parser.add_argument(
        "file_path",
        help="Path to the paper-like *_notch58_62.mat file",
    )
    parser.add_argument("--loc-file", help="Path to matching *_xslocs.mat")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--n-channels", type=int, default=DEFAULT_N_CHANNELS)
    parser.add_argument(
        "--scale",
        type=float,
        help="Initial display scale. By default it is estimated automatically.",
    )
    args = parser.parse_args()

    file_path = Path(args.file_path).resolve()
    subject = subject_from_path(file_path)
    loc_path = Path(args.loc_file).resolve() if args.loc_file else default_loc_path(file_path, subject)

    data, stim, srate, elcode, locs, notch_band, filter_order = load_data_and_localisation(
        file_path, loc_path
    )
    n_ch, n_times = data.shape
    total_duration = n_times / srate

    channel_names = [f"E{i + 1}" for i in range(n_ch)]
    areas = [AREA_LABELS.get(int(code), f"Unknown code {int(code)}") for code in elcode]
    rois = [ROI_BY_AREA.get(area, "Excluded") for area in areas]
    auto_scale = estimate_display_scale(data, srate)
    scale0 = float(args.scale) if args.scale is not None else auto_scale
    if not np.isfinite(scale0) or scale0 <= 0:
        raise ValueError("Display scale must be a positive finite number.")

    out_path = file_path.with_name(f"{subject}_bad_channels.txt")
    bad_channels = set()
    if out_path.exists():
        bad_channels = {
            line.strip() for line in out_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    print("Subject:", subject)
    print("Raw file:", file_path)
    print("Localisation file:", loc_path)
    print("Data shape:", data.shape, "(channels x time)")
    print("Sampling rate:", srate, "Hz")
    print("Verified mapping: E1 = data[:, 0] = ElectrodeIndex 1")
    paper_like_filter = (
        notch_band is not None
        and notch_band.shape == (2,)
        and np.allclose(notch_band, [58.0, 62.0])
        and filter_order == 3
    )
    if paper_like_filter:
        filter_label = "Butterworth order 3, 58-62 Hz"
        print("Verified display input filter:", filter_label)
    else:
        filter_label = "unverified"
        print("WARNING: this is not a verified 58-62 Hz, order-3 filtered file.")
        print("Create it from the original recording with save_notched_mat.py.")
    print("Automatic display scale:", f"{auto_scale:.6g}")
    print("Initial display scale:", f"{scale0:.6g}")
    print("Existing bad channels:", ", ".join(sorted(bad_channels)) or "none")

    n_channels0 = int(np.clip(args.n_channels, 1, n_ch))
    duration0 = float(np.clip(args.duration, 0.5, total_duration))
    state = {
        "start_s": 0.0,
        "first_ch": 0,
        "n_channels": n_channels0,
        "scale": scale0,
        "duration": duration0,
    }

    fig, ax = plt.subplots(figsize=(16, 9))
    plt.subplots_adjust(left=0.31, right=0.98, top=0.90, bottom=0.30)

    def visible_channel_from_y(y):
        first = int(state["first_ch"])
        last = min(n_ch, first + int(state["n_channels"]))
        n_visible = last - first
        if y is None or n_visible <= 0:
            return None
        offset = int(round(y))
        if offset < 0 or offset >= n_visible:
            return None
        return first + (n_visible - 1 - offset)

    def draw():
        ax.clear()
        start = int(state["start_s"] * srate)
        stop = int(min(n_times, start + state["duration"] * srate))
        first = int(state["first_ch"])
        last = min(n_ch, first + int(state["n_channels"]))

        times = np.arange(start, stop) / srate
        visible = data[first:last, start:stop]
        offsets = np.arange(last - first)[::-1]

        for local_index, channel_data in enumerate(visible):
            channel_index = first + local_index
            channel_name = channel_names[channel_index]
            is_bad = channel_name in bad_channels
            display_data = channel_data - np.nanmedian(channel_data)
            ax.plot(
                times,
                display_data / state["scale"] + offsets[local_index],
                color="tab:red" if is_bad else "black",
                linewidth=1.2 if is_bad else 0.7,
            )

        labels = []
        colors = []
        for channel_index in range(first, last):
            name = channel_names[channel_index]
            suffix = " | BAD" if name in bad_channels else ""
            labels.append(f"{name} | {areas[channel_index]} | {rois[channel_index]}{suffix}")
            colors.append("tab:red" if name in bad_channels else "black")

        ax.set_yticks(offsets)
        ax.set_yticklabels(labels, fontsize=8)
        for tick, color in zip(ax.get_yticklabels(), colors):
            tick.set_color(color)

        ax.set_xlabel("Time (s)")
        ax.set_title(
            f"Subject {subject} | click trace/label to toggle bad | "
            f"bads={len(bad_channels)} | scale={state['scale']:.3g} | filter={filter_label}"
        )
        ax.grid(True, axis="x", alpha=0.25)
        if len(times):
            ax.set_xlim(times[0], times[-1])
        ax.set_ylim(-1, max(1, len(offsets)))
        fig.canvas.draw_idle()

    ax_time = plt.axes([0.31, 0.205, 0.60, 0.03])
    ax_first = plt.axes([0.31, 0.160, 0.60, 0.03])
    ax_count = plt.axes([0.31, 0.115, 0.60, 0.03])
    ax_scale = plt.axes([0.31, 0.070, 0.60, 0.03])

    time_slider = Slider(ax_time, "time", 0, max(0.0, total_duration - duration0), valinit=0, valstep=0.1)
    first_slider = Slider(ax_first, "first ch", 0, max(0, n_ch - 1), valinit=0, valstep=1)
    count_slider = Slider(ax_count, "n ch", 1, n_ch, valinit=n_channels0, valstep=1)
    slider_min = min(auto_scale / 20, scale0 / 2)
    slider_max = max(auto_scale * 20, scale0 * 2)
    scale_slider = Slider(
        ax_scale,
        "scale",
        slider_min,
        slider_max,
        valinit=scale0,
    )

    def update(_):
        state["start_s"] = float(time_slider.val)
        state["first_ch"] = int(first_slider.val)
        state["n_channels"] = int(count_slider.val)
        state["scale"] = float(scale_slider.val)
        max_first = max(0, n_ch - state["n_channels"])
        if state["first_ch"] > max_first:
            first_slider.set_val(max_first)
            return
        draw()

    for slider in (time_slider, first_slider, count_slider, scale_slider):
        slider.on_changed(update)

    def on_click(event):
        if event.inaxes != ax:
            return
        channel_index = visible_channel_from_y(event.ydata)
        if channel_index is None:
            return
        name = channel_names[channel_index]
        if name in bad_channels:
            bad_channels.remove(name)
            print("Unmarked bad:", name)
        else:
            bad_channels.add(name)
            print("Marked bad:", name, "|", areas[channel_index], "|", rois[channel_index])
        draw()

    fig.canvas.mpl_connect("button_press_event", on_click)

    ax_prev = plt.axes([0.31, 0.015, 0.09, 0.035])
    ax_next = plt.axes([0.41, 0.015, 0.09, 0.035])
    ax_save = plt.axes([0.52, 0.015, 0.13, 0.035])
    ax_auto = plt.axes([0.66, 0.015, 0.13, 0.035])
    prev_button = Button(ax_prev, "Prev")
    next_button = Button(ax_next, "Next")
    save_button = Button(ax_save, "Save bads")
    auto_button = Button(ax_auto, "Auto scale")

    prev_button.on_clicked(
        lambda _: time_slider.set_val(max(0, time_slider.val - state["duration"]))
    )
    next_button.on_clicked(
        lambda _: time_slider.set_val(
            min(max(0.0, total_duration - state["duration"]), time_slider.val + state["duration"])
        )
    )

    def save_bads(_):
        ordered = sorted(bad_channels, key=lambda name: int(name[1:]))
        out_path.write_text("\n".join(ordered) + ("\n" if ordered else ""), encoding="utf-8")
        print("Saved bad channels to:", out_path)
        print("Saved bads:", ", ".join(ordered) or "none")

    save_button.on_clicked(save_bads)
    auto_button.on_clicked(lambda _: scale_slider.set_val(auto_scale))
    draw()
    plt.show()


if __name__ == "__main__":
    main()
