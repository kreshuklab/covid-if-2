# based on:
# https://github.com/hci-unihd/batchlib/blob/e7e4ce34228346c5dcf1d1f01c512d5fad5ba5c1/batchlib/util/plate_visualizations.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Wedge
from mpl_toolkits.axes_grid1 import make_axes_locatable

from plate_utils import to_position

ROW_LETTERS = np.array(list("ABCDEFGH"))
LETTER_TO_ROW = {letter: i for i, letter in enumerate(ROW_LETTERS)}
ABBREVIATIONS = {
    "Wildtype - Giantin": "WT",
    "Delta - LCK": "Delta",
    "Omicron BA.1 - H2A": "Om"
}


def get_scores(table, use_spike_average):
    table = table.dropna()
    well_names = pd.unique(table["well"])

    def get_score(name, pattern):
        row_selector = (table["well"] == name) & (table["pattern"] == pattern)
        if row_selector.sum() != 1:
            return 0.0
        else:
            return table[row_selector]["score"].item()

    def get_max_score(name, patterns):
        row_selectors = (table["well"] == name) & table["pattern"].isin(patterns)
        if row_selectors.sum() == 0:
            return 0.0, "NaN"
        rows = table[row_selectors]
        max_row_id = np.argmax(rows["score"].values)
        return rows.iloc[max_row_id]["score"], ABBREVIATIONS[rows.iloc[max_row_id]["pattern"]]

    spike_patterns = ["Wildtype - Giantin", "Delta - LCK", "Omicron BA.1 - H2A"]
    if use_spike_average:
        spike_scores = {name: get_score(name, "Spike") for name in well_names}
        spike_types = {name: None for name in well_names}
    else:
        scores_and_types = [get_max_score(name, spike_patterns) for name in well_names]
        spike_scores = {name: sat[0] for name, sat in zip(well_names, scores_and_types)}
        spike_types = {name: sat[1] for name, sat in zip(well_names, scores_and_types)}

    ncap_scores = {name: get_score(name, "Nucleocapsid - 3xNLS") for name in well_names}

    return spike_scores, ncap_scores, spike_types


def get_qc_passed(table):
    table = table.dropna()
    well_names = pd.unique(table["well"])
    qc_passed = {
        name: table[table["well"] == name]["min_num_for_qc"].values.all() for name in well_names
    }
    return qc_passed


def plate_overview_plot(
    table, save_path=None, figsize=(14, 8), plate_name=None,
    use_spike_average=True, mark_outliers=False,
):
    radius = 0.5

    fig, ax = plt.subplots(figsize=figsize)

    spike_scores, ncap_scores, spike_types = get_scores(table, use_spike_average=use_spike_average)
    qc_passed = get_qc_passed(table)

    patches, patch_values = [], []
    qc_patches = []

    for well_name, spike_score in spike_scores.items():

        well_position = to_position(well_name)
        # map "A01" to (0, 0)
        well_position = (well_position[0] - 1, well_position[1])
        # Map to the matplotlib coordinate space
        center = (well_position[0], 7 - well_position[1])

        # TODO well level QC for min number of cells per pattern

        spike_wedge = Wedge(center, radius, 1, 179)
        patches.append(spike_wedge)
        patch_values.append(spike_score)
        center_text = (center[0], center[1] + 0.25)
        spike_type = spike_types[well_name]
        if spike_type is None:
            t = plt.annotate(f"{spike_score:.2f}", center_text, ha="center", va="center")
        else:
            t = plt.annotate(f"{spike_score:.2f} {spike_type}", center_text, ha="center", va="center")
        t.set_bbox(dict(edgecolor="white", facecolor="white", alpha=0.25))

        ncap_score = ncap_scores[well_name]
        ncap_wedge = Wedge(center, radius, 181, 359)
        patches.append(ncap_wedge)
        patch_values.append(ncap_score)
        center_text = (center[0], center[1] - 0.25)
        t = plt.annotate(f"{ncap_score:.2f}", center_text, ha="center", va="center")
        t.set_bbox(dict(edgecolor="white", facecolor="white", alpha=0.25))

        if mark_outliers and not qc_passed[well_name]:
            rect_center = (center[0] - radius, center[1] - radius)
            qc_rect = Rectangle(rect_center, 2 * radius, 2 * radius, color="red")
            qc_patches.append(qc_rect)

    coll = PatchCollection(patches + qc_patches)
    coll.set_array(np.array(patch_values))
    coll.set_clim(0.0, 2.6)
    coll.set_cmap("seismic")

    ax.add_collection(coll)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xticks(np.arange(12), np.arange(1, 13))
    plt.yticks(np.arange(len(ROW_LETTERS)), reversed(ROW_LETTERS))
    plt.xlim(-0.7, 11.7)
    plt.ylim(-0.7, 7.7)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    fig.colorbar(coll, cax=cax)

    if plate_name is None:
        ax.set_title("Scores in circle: Top=Spike, Bottom=Ncapsid")
    else:
        ax.set_title(f"Plate: {plate_name}, Scores in circle: Top=Spike, Bottom=Ncapsid")

    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def main():
    tab = pd.read_excel("./analysis_results/230131_ns_plate_10c1/230131_ns_plate_10c1.xlsx")
    plate_overview_plot(tab, plate_name="230131_ns_plate_10c1", use_spike_average=False)


if __name__ == "__main__":
    main()
