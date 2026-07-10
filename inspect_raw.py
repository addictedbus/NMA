"""
Run from VS Code terminal, not from a notebook cell.

Install dependencies once:
    pip install numpy scipy matplotlib

Usage:
    python inspect_raw.py "<path-to-subject>\aa_faceshouses.mat"

Optional:
    python inspect_raw.py "<path-to-subject>\aa_faceshouses.mat" --duration 8 --n-channels 30 --scale 3000000

Click a channel trace or label area to toggle it as bad. Bad channels are shown in red.
Click "Save bads" to write a text file with selected bad channel names.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.io import loadmat

DEFAULT_DURATION = 8.0
DEFAULT_N_CHANNELS = 30
DEFAULT_SCALE = 3_000_000.0


def subject_from_path(file_path):
    path = Path(file_path)
    return path.stem.replace("_faceshouses", "")


def load_faceshouses(file_path):
    mat = loadmat(file_path, squeeze_me=True)
    data = np.asarray(mat["data"], dtype=float).T  # channels x time
    stim = np.asarray(mat["stim"])
    srate = float(mat["srate"])
    return data, stim, srate


def main():
    parser = argparse.ArgumentParser(description="Interactive raw viewer with bad-channel marking.")
    parser.add_argument("file_path", help="Path to *_faceshouses.mat")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="Visible time window in seconds.")
    parser.add_argument("--n-channels", type=int, default=DEFAULT_N_CHANNELS, help="Initial number of channels shown.")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="Initial amplitude scale. Larger value makes traces smaller.")
    args = parser.parse_args()

    subject = subject_from_path(args.file_path)
    data, stim, srate = load_faceshouses(args.file_path)
    n_ch, n_times = data.shape
    total_duration = n_times / srate
    ch_names = [f"EEG{i}" for i in range(n_ch)]

    n_channels0 = int(np.clip(args.n_channels, 1, n_ch))
    duration0 = float(np.clip(args.duration, 0.5, total_duration))
    scale0 = float(args.scale)
    bad_channels = set()

    out_path = Path(args.file_path).with_name(f"{subject}_bad_channels.txt")

    print("Subject:", subject)
    print("File:", args.file_path)
    print("Data shape:", data.shape, "(channels x time)")
    print("Sampling rate:", srate, "Hz")
    print("Duration:", total_duration, "s")
    print("Data min/max:", np.nanmin(data), np.nanmax(data))
    print("Median channel std:", np.median(np.std(data, axis=1)))
    print("Click a channel to mark/unmark it as bad. Bad channels are red.")

    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.30)

    state = {
        "start_s": 0.0,
        "first_ch": 0,
        "n_channels": n_channels0,
        "scale": scale0,
        "duration": duration0,
    }

    def visible_channel_from_y(y):
        first = int(state["first_ch"])
        count = int(state["n_channels"])
        last = min(n_ch, first + count)
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
        count = int(state["n_channels"])
        last = min(n_ch, first + count)

        t = np.arange(start, stop) / srate
        visible = data[first:last, start:stop]
        offsets = np.arange(last - first)[::-1]

        for i, ch_data in enumerate(visible):
            ch_idx = first + i
            name = ch_names[ch_idx]
            color = "tab:red" if name in bad_channels else "black"
            linewidth = 1.2 if name in bad_channels else 0.7
            y = ch_data / state["scale"] + offsets[i]
            ax.plot(t, y, color=color, linewidth=linewidth)

        tick_colors = []
        labels = []
        for ch in range(first, last):
            name = ch_names[ch]
            labels.append(name + ("  BAD" if name in bad_channels else ""))
            tick_colors.append("tab:red" if name in bad_channels else "black")

        ax.set_yticks(offsets)
        ax.set_yticklabels(labels)
        for tick, color in zip(ax.get_yticklabels(), tick_colors):
            tick.set_color(color)

        ax.set_xlabel("Time (s)")
        ax.set_title(
            f"Subject {subject}: click channels to toggle bad | "
            f"bads={len(bad_channels)} | scale={state['scale']:.3g}"
        )
        ax.grid(True, axis="x", alpha=0.25)
        if len(t):
            ax.set_xlim(t[0], t[-1])
        ax.set_ylim(-1, max(1, len(offsets)))
        fig.canvas.draw_idle()

    ax_time = plt.axes([0.14, 0.205, 0.76, 0.03])
    ax_first = plt.axes([0.14, 0.160, 0.76, 0.03])
    ax_count = plt.axes([0.14, 0.115, 0.76, 0.03])
    ax_scale = plt.axes([0.14, 0.070, 0.76, 0.03])

    time_slider = Slider(ax_time, "time", 0, max(0.0, total_duration - duration0), valinit=0, valstep=0.1)
    first_slider = Slider(ax_first, "first ch", 0, max(0, n_ch - 1), valinit=0, valstep=1)
    count_slider = Slider(ax_count, "n ch", 1, n_ch, valinit=n_channels0, valstep=1)
    scale_slider = Slider(ax_scale, "scale", 100_000, 20_000_000, valinit=scale0, valstep=100_000)

    def update(_):
        state["start_s"] = float(time_slider.val)
        state["first_ch"] = int(first_slider.val)
        state["n_channels"] = int(count_slider.val)
        state["scale"] = float(scale_slider.val)

        max_first = max(0, n_ch - state["n_channels"])
        if state["first_ch"] > max_first:
            state["first_ch"] = max_first
            first_slider.set_val(max_first)
            return
        draw()

    for slider in [time_slider, first_slider, count_slider, scale_slider]:
        slider.on_changed(update)

    def on_click(event):
        if event.inaxes != ax:
            return
        ch_idx = visible_channel_from_y(event.ydata)
        if ch_idx is None:
            return
        name = ch_names[ch_idx]
        if name in bad_channels:
            bad_channels.remove(name)
            print("Unmarked bad:", name)
        else:
            bad_channels.add(name)
            print("Marked bad:", name)
        print("Current bads:", ", ".join(sorted(bad_channels)) if bad_channels else "none")
        draw()

    fig.canvas.mpl_connect("button_press_event", on_click)

    ax_prev = plt.axes([0.14, 0.015, 0.10, 0.035])
    ax_next = plt.axes([0.25, 0.015, 0.10, 0.035])
    ax_save = plt.axes([0.37, 0.015, 0.14, 0.035])
    prev_button = Button(ax_prev, "Prev")
    next_button = Button(ax_next, "Next")
    save_button = Button(ax_save, "Save bads")

    def prev_page(_):
        time_slider.set_val(max(0, time_slider.val - state["duration"]))

    def next_page(_):
        time_slider.set_val(min(max(0.0, total_duration - state["duration"]), time_slider.val + state["duration"]))

    def save_bads(_):
        ordered = sorted(bad_channels, key=lambda name: int(name.replace("EEG", "")))
        out_path.write_text("\n".join(ordered) + ("\n" if ordered else ""), encoding="utf-8")
        print("Saved bad channels to:", out_path)
        print("Saved bads:", ", ".join(ordered) if ordered else "none")

    prev_button.on_clicked(prev_page)
    next_button.on_clicked(next_page)
    save_button.on_clicked(save_bads)

    draw()
    plt.show()


if __name__ == "__main__":
    main()
