# based on:
# https://github.com/hci-unihd/batchlib/blob/e7e4ce34228346c5dcf1d1f01c512d5fad5ba5c1/batchlib/util/plate_visualizations.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge
from mpl_toolkits.axes_grid1 import make_axes_locatable

from plate_utils import to_position

ROW_LETTERS = np.array(list("ABCDEFGH"))
LETTER_TO_ROW = {letter: i for i, letter in enumerate(ROW_LETTERS)}


def get_scores(table):
    table = table.dropna()
    well_names = pd.unique(table["well"])
    spike_scores = {name: table[
        (table["well"] == name) & (table["pattern"] == "Spike")
    ]["score"].item() for name in well_names}
    ncap_scores = {name: table[
        (table["well"] == name) & (table["pattern"] == "Nucleocapsid - 3xNLS")
    ]["score"].item() for name in well_names}
    return spike_scores, ncap_scores


def plate_overview_plot(table, save_path=None, figsize=(14, 8)):
    radius = 0.5

    fig, ax = plt.subplots(figsize=figsize)

    spike_scores, ncap_scores = get_scores(table)
    patches, patch_values = [], []

    for well_name, spike_score in spike_scores.items():
        well_position = to_position(well_name)
        # map "A01" to (0, 0)
        well_position = (well_position[0] - 1, well_position[1])
        # Map to the matplotlib coordinate space
        center = (well_position[0], 7 - well_position[1])

        spike_wedge = Wedge(center, radius, 1, 179)
        patches.append(spike_wedge)
        patch_values.append(spike_score)
        center_text = (center[0], center[1] + 0.25)
        t = plt.annotate(f"{spike_score:.2f}", center_text, ha="center", va="center")
        t.set_bbox(dict(edgecolor="white", facecolor="white", alpha=0.25))

        ncap_score = ncap_scores[well_name]
        ncap_wedge = Wedge(center, radius, 181, 359)
        patches.append(ncap_wedge)
        patch_values.append(ncap_score)
        center_text = (center[0], center[1] - 0.25)
        t = plt.annotate(f"{ncap_score:.2f}", center_text, ha="center", va="center")
        t.set_bbox(dict(edgecolor="white", facecolor="white", alpha=0.25))

    coll = PatchCollection(patches)
    coll.set_array(np.array(patch_values))

    ax.add_collection(coll)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xticks(np.arange(12), np.arange(1, 13))
    plt.yticks(np.arange(len(ROW_LETTERS)), reversed(ROW_LETTERS))
    plt.xlim(-0.7, 11.7)
    plt.ylim(-0.7, 7.7)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    fig.colorbar(coll, cax=cax)

    ax.set_title("Scores in circle: Top=Spike, Bottom=Ncapsid")

    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def main():
    tab = pd.read_excel("./analysis_results/230131_ns_plate_10c1/230131_ns_plate_10c1.xlsx")
    plate_overview_plot(tab)


if __name__ == "__main__":
    main()
