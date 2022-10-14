import os
import string
from glob import glob

import mobie
import mobie.htm as htm
import numpy as np
import pandas as pd

from czifile import CziFile
from tqdm import tqdm

INPUT_ROOT = "/g/kreshuk/data/covid-if-2/from_nuno"
OUTPUT_ROOT = "/scratch/pape/covid-if-2/data"

# TODO read channel order from file
CHANNEL_ORDER = {0: "marker", 1: "nuclei", 2: "serum"}


def read_czi(path):
    data = {}
    samples = []
    with CziFile(path, "r") as f:
        for block in f.subblocks():
            sample = [d.start for d in block.dimension_entries if d.dimension == "S"][0]
            channel = [d.start for d in block.dimension_entries if d.dimension == "C"][0]
            samples.append(sample)
            data[channel] = block.data().squeeze()
    assert len(set(samples)) == 1
    assert len(data) == 3
    data = np.concatenate([data[channel][None] for channel in CHANNEL_ORDER], axis=0)
    return data


def convert_images(in_folder, ds_folder, pos_to_pattern):
    sources = mobie.metadata.read_dataset_metadata(ds_folder).get("sources", {})

    ds_name = os.path.basename(ds_folder)
    out_root = os.path.split(ds_folder)[0]

    resolution = [1.0, 1.0]
    chunks = (1024, 1024)
    scale_factors = [[2, 2], [2, 2]]

    pattern_name = os.path.basename(in_folder).replace(" ", "_")
    image_files = glob(os.path.join(in_folder, "*.czi"))

    for image_path in tqdm(image_files, desc="Convert images"):
        fname = os.path.splitext(os.path.basename(image_path))[0]
        pos = fname.split("_")[-1]
        pos_to_pattern[pos] = pattern_name
        if all(f"{channel_name}_{pos}" in sources for channel_name in CHANNEL_ORDER.values()):
            continue
        image_data = read_czi(image_path)
        for channel, channel_name in CHANNEL_ORDER.items():
            source_name = f"{channel_name}_{pos}"
            tmp_folder = f"tmps/tmp_{source_name}"
            mobie.add_image(image_data[channel], None, out_root, ds_name, source_name,
                            resolution=resolution, view={}, tmp_folder=tmp_folder,
                            scale_factors=scale_factors, chunks=chunks,
                            unit="pixel", file_format="ome.zarr")

    return pos_to_pattern


def to_site_name(source_name, prefix):
    pos = source_name.split("_")[1]
    pos_id = int(pos[1:]) - 1
    well_id = pos_id // 9
    pos_in_well = pos_id % 9
    # this is hardcoded to the current format for the "Markers_New" data
    return f"A{well_id:02}_{pos_in_well}"


def to_well_name(site_name):
    return site_name.split("_")[0]


def to_position(well_name):
    r, c = well_name[0], well_name[1:]
    r = string.ascii_uppercase.index(r)
    return [int(c), r]


def get_tables(pos_to_pattern):

    region_ids = []
    patterns = []
    for pos, pattern in pos_to_pattern.items():
        source_name = f"blub_{pos}"
        site_name = to_site_name(source_name, "blub")
        region_ids.append(site_name)
        patterns.append(pattern)

    site_table = pd.DataFrame.from_dict({
        "region_id": region_ids, "pattern": patterns
    })

    region_ids_well = []
    patterns_well = []
    for site_name, pattern in zip(region_ids, patterns):
        well_name = to_well_name(site_name)
        if well_name in region_ids_well:
            continue
        region_ids_well.append(well_name)
        patterns_well.append(pattern)

    well_table = pd.DataFrame.from_dict({
        "region_id": region_ids_well, "pattern": patterns_well
    })

    return site_table, well_table


def add_grid_view(ds_folder, site_table, well_table):

    source_prefixes = list(CHANNEL_ORDER.values())
    source_types = len(CHANNEL_ORDER) * ["image"]

    contrast_limits = {
        name: htm.compute_contrast_limits(
            name, ds_folder, lower_percentile=4, upper_percentile=96, n_threads=16
        )
        for name in source_prefixes
    }
    # contrast_limits = {name: [0, 0] for name in source_prefixes}
    colors = {"nuclei": "blue", "serum": "green", "marker": "red"}
    source_settings = [
        {"color": colors[name], "contrastLimits": contrast_limits[name], "visible": True}
        for name in source_prefixes
    ]

    htm.add_plate_grid_view(
        ds_folder, view_name="default", source_prefixes=source_prefixes, source_types=source_types,
        source_settings=source_settings, source_name_to_site_name=to_site_name,
        site_name_to_well_name=to_well_name, well_to_position=to_position,
        site_table=site_table, well_table=well_table,
        sites_visible=False, menu_name="bookmark"
    )


def convert_to_mobie(folder_name):
    in_root = os.path.join(INPUT_ROOT, folder_name)

    if not mobie.metadata.project_exists(OUTPUT_ROOT):
        mobie.metadata.create_project_metadata(OUTPUT_ROOT)

    pos_to_pattern = {}
    ds_folder = os.path.join(OUTPUT_ROOT, folder_name.lower())
    input_folders = glob(os.path.join(in_root, "*"))
    for in_folder in input_folders:
        if not os.path.isdir(in_folder):
            assert False, in_folder
        pos_to_pattern = convert_images(in_folder, ds_folder, pos_to_pattern)

    site_table, well_table = get_tables(pos_to_pattern)
    add_grid_view(ds_folder, site_table, well_table)


def main():
    folder_name = "Markers_New"
    convert_to_mobie(folder_name)


if __name__ == "__main__":
    main()
