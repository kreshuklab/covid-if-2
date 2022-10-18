# find the cells without staining:
# - go over the nuclei and compare their mean intensity in the marker channel to the background
# - if the median expression is smaller than (bg_median + n * bg_mad) than classify it as non-expressed
# - (n to be determined)
import os

import pandas as pd

OUTPUT_ROOT = "/scratch/pape/covid-if-2/data"


def find_stained_cells_image(ds_folder, position, bg_median, bg_mad):
    pos_name = position.capitalize()
    table_path = os.path.join(
        ds_folder, "tables", f"cell-segmentation_{pos_name}", "statistics_nucleus-segmentation.tsv"
    )
    table = pd.read_csv(table_path, sep="\t")

    n = 3
    stain_threshold = bg_median + n * bg_mad
    nuc_intensity = table["marker_median"]
    is_stained = nuc_intensity > stain_threshold
    print("Position:", position, ":", is_stained.sum(), "of", is_stained.size, "cells are stained")

    # add to the default table
    default_table_path = os.path.join(
        ds_folder, "tables", f"cell-segmentation_{pos_name}", "default.tsv"
    )
    default_table = pd.read_csv(default_table_path, sep="\t")
    default_table["is_stained"] = is_stained
    default_table.to_csv(default_table_path, sep="\t", index=False)


def find_stained_cells(ds_name):
    ds_folder = os.path.join(OUTPUT_ROOT, ds_name)
    site_table = pd.read_csv(os.path.join(ds_folder, "tables", "sites", "default.tsv"), sep="\t")
    bg_table = pd.read_csv(os.path.join(ds_folder, "tables", "sites", "bg_stats.tsv"), sep="\t")
    bg_medians, bg_mads = bg_table["marker_median"].values, bg_table["marker_mad"].values

    positions = site_table["position"].values
    assert len(positions) == len(bg_medians)

    # debugging
    for position, bg_median, bg_mad in zip(positions, bg_medians, bg_mads):
        find_stained_cells_image(ds_folder, position, bg_median, bg_mad)


def main():
    find_stained_cells("markers_new")


if __name__ == "__main__":
    main()
