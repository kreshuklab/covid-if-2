import os

import numpy as np
import pandas as pd

OUTPUT_ROOT = "/scratch/pape/covid-if-2/data"
SHAPE = (3008, 4096)


def find_stained_cells_image(ds_folder, position, bg_median, bg_mad):
    pos_name = position.capitalize()
    table_path = os.path.join(
        ds_folder, "tables", f"cell-segmentation_{pos_name}", "statistics_nucleus-segmentation.tsv"
    )
    intensity_table = pd.read_csv(table_path, sep="\t")
    default_table_path = os.path.join(
        ds_folder, "tables", f"cell-segmentation_{pos_name}", "default.tsv"
    )
    table = pd.read_csv(default_table_path, sep="\t")
    if len(intensity_table) != len(table):
        assert len(intensity_table) > len(table)
        intensity_table = intensity_table.iloc[
            np.isin(intensity_table["label_id"].values, table["label_id"].values)
        ]
    assert len(table) == len(intensity_table), f"{len(table)} == {len(intensity_table)}"

    # find the cells without staining:
    # - go over the nuclei and compare their mean intensity in the marker channel to the background
    # - if the median expression is smaller than (bg_median + n * bg_mad) than classify it as non-expressed
    n = 3
    stain_threshold = bg_median + n * bg_mad
    nuc_intensity = intensity_table["marker_median"].values
    is_stained = nuc_intensity > stain_threshold
    print("Position:", position, ":", is_stained.sum(), "of", is_stained.size, "cells are stained")

    # filter the cells with small / large bounding boxes
    # (10% of largest, 10% of smallest)
    cell_bb_shape = np.concatenate([(table["bb_max_y"] - table["bb_min_y"]).values[:, None],
                                    (table["bb_max_x"] - table["bb_min_x"]).values[:, None]], axis=1)
    cell_bb_size = np.prod(cell_bb_shape, axis=1)
    min_size = np.percentile(cell_bb_size, 10)
    max_size = np.percentile(cell_bb_size, 90)
    is_size_conforming = np.logical_and(cell_bb_size > min_size, cell_bb_size < max_size)
    print(
        "Position:", position, ":", is_size_conforming.sum(), "of", is_size_conforming.size, "cells are size conforming"
    )

    # filter cells that touch the boundary
    is_lower_boundary = np.logical_or(np.isclose(table["bb_min_y"].values, 0), np.isclose(table["bb_min_x"].values, 0))
    is_upper_boundary = np.logical_or(
        np.isclose(table["bb_max_y"].values, SHAPE[0]), np.isclose(table["bb_max_x"].values, SHAPE[1])
    )
    is_boundary = np.logical_or(is_lower_boundary, is_upper_boundary)
    is_inner = np.logical_not(is_boundary)
    print("Position:", position, ":", is_inner.sum(), "of", is_inner.size, "cells are not on the boundary")

    is_train = np.sum(np.concatenate(
        [is_stained[:, None], is_size_conforming[:, None], is_inner[:, None]], axis=1
    ), axis=1) == 3
    print("Position:", position, ":", is_train.sum(), "of", is_train.size, "cells will be used for training")

    # add to the default table
    table["is_stained"] = is_stained
    table["is_train"] = is_train
    table.to_csv(default_table_path, sep="\t", index=False)


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
