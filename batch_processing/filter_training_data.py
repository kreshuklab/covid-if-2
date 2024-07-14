import argparse
import os

import mobie
import numpy as np
import pandas as pd

from tqdm import tqdm
from plate_utils import read_plate_config, OUTPUT_ROOT
SHAPE = (3008, 4096)


def filter_training_data_position(ds_folder, position, metadata):
    view = metadata["views"]["segmentations"]
    displays = view["sourceDisplays"]
    grid_layout = None
    for display in displays:
        type_, values = next(iter(display.items()))
        if type_ == "regionDisplay" and values["name"] == "sites":
            grid_layout = values["sources"]
    assert grid_layout is not None

    pos_sources = grid_layout[position]
    seg_source = [source for source in pos_sources if "cell-segmentation" in source]
    assert len(seg_source) == 1
    seg_source = seg_source[0]

    sources = metadata["sources"]
    table_folder = os.path.join(
        ds_folder, sources[seg_source]["segmentation"]["tableData"]["tsv"]["relativePath"]
    )
    assert os.path.exists(table_folder)

    intensity_table_path = os.path.join(table_folder, "statistics_nucleus-segmentation.tsv")
    intensity_table = pd.read_csv(intensity_table_path, sep="\t")

    default_table_path = os.path.join(table_folder, "default.tsv")
    table = pd.read_csv(default_table_path, sep="\t")
    assert (intensity_table["label_id"] == table["label_id"]).all()

    # filter the cells with a very low marker intensity
    stain_threshold = 100
    nuc_intensity = intensity_table["marker_median"].values
    is_stained = nuc_intensity > stain_threshold
    print("Position:", position, ":", is_stained.sum(), "of", is_stained.size, "cells are stained")

    # filter the cells with small / large bounding boxes
    # (5% of largest, 10% of smallest)
    cell_bb_shape = np.concatenate([(table["bb_max_y"] - table["bb_min_y"]).values[:, None],
                                    (table["bb_max_x"] - table["bb_min_x"]).values[:, None]], axis=1)
    cell_bb_size = np.prod(cell_bb_shape, axis=1)
    min_size = np.percentile(cell_bb_size, 10)
    max_size = np.percentile(cell_bb_size, 95)
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
    table["is_train"] = is_train
    table.to_csv(default_table_path, sep="\t", index=False)


def filter_training_data(ds_name):
    ds_folder = os.path.join(OUTPUT_ROOT, ds_name)
    metadata = mobie.metadata.read_dataset_metadata(ds_folder)
    site_table = pd.read_csv(os.path.join(ds_folder, "tables", "sites", "default.tsv"), sep="\t")
    positions = site_table["region_id"].values
    for position in tqdm(positions):
        filter_training_data_position(ds_folder, position, metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")  # e.g. "./plate_configs/mix_wt_alpha_control.json"
    args = parser.parse_args()
    plate_config = read_plate_config(args.config_file)
    folder_name = os.path.basename(plate_config.folder).lower()
    filter_training_data(folder_name)


if __name__ == "__main__":
    main()
