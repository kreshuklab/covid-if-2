import argparse
import os
from glob import glob

import mobie
import mobie.htm as htm
import numpy as np
import pandas as pd

from czifile import CziFile
from tqdm import tqdm

from plate_utils import to_well_name, read_plate_config

INPUT_ROOT = "/g/kreshuk/data/covid-if-2/from_nuno"
# OUTPUT_ROOT = "/scratch/pape/covid-if-2/data"
OUTPUT_ROOT = "/g/kreshuk/data/covid-if-2/from_nuno/mobie-tmp/data"


def read_czi(path, channel_order):
    data = {}
    samples = []
    with CziFile(path, "r") as f:
        for block in f.subblocks():
            sample = [d.start for d in block.dimension_entries if d.dimension == "S"][0]
            channel = [d.start for d in block.dimension_entries if d.dimension == "C"][0]
            samples.append(sample)
            data[channel] = block.data().squeeze()
    assert len(set(samples)) == 1
    assert len(data) == len(channel_order), f"{len(data)}, {channel_order}, {path}"
    data = np.concatenate([data[channel][None] for channel in channel_order], axis=0)
    return data


def convert_images(in_folder, ds_folder, pos_to_pattern, channel_order):
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
        # breakpoint()
        # print(pos, fname, pattern_name)
        if all(f"{channel_name}_{pos}" in sources for channel_name in channel_order.values()):
            continue
        image_data = read_czi(image_path, channel_order)
        for channel, channel_name in channel_order.items():
            source_name = f"{channel_name}_{pos}"
            tmp_folder = f"tmps/tmp_{source_name}"
            mobie.add_image(image_data[channel], None, out_root, ds_name, source_name,
                            resolution=resolution, view={}, tmp_folder=tmp_folder,
                            scale_factors=scale_factors, chunks=chunks,
                            unit="pixel", file_format="ome.zarr")

    return pos_to_pattern


def get_tables(pos_to_pattern, table_root, to_site_name):

    region_ids = []
    patterns = []
    positions = []
    for pos, pattern in pos_to_pattern.items():
        source_name = f"blub_{pos}"
        site_name = to_site_name(source_name, "blub")
        region_ids.append(site_name)
        patterns.append(pattern)
        positions.append(pos)

    site_table = pd.DataFrame.from_dict({
        "region_id": region_ids, "pattern": patterns, "position": positions
    })
    site_table_path = os.path.join(table_root, "sites", "default.tsv")
    os.makedirs(os.path.join(table_root, "sites"), exist_ok=True)
    site_table = site_table.sort_values("region_id")
    site_table.to_csv(site_table_path, sep="\t", index=False)

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
    well_table_path = os.path.join(table_root, "wells", "default.tsv")
    well_table = well_table.sort_values("region_id")
    os.makedirs(os.path.join(table_root, "wells"), exist_ok=True)
    well_table.to_csv(well_table_path, sep="\t", index=False)

    return site_table, well_table


def add_grid_view(ds_folder, site_table, well_table, channel_order, channel_colors, to_site_name):

    source_prefixes = list(channel_order.values())
    source_types = len(channel_order) * ["image"]

    contrast_limits = {
        name: htm.compute_contrast_limits(
            name, ds_folder, lower_percentile=4, upper_percentile=96, n_threads=16
        )
        for name in source_prefixes
    }
    # contrast_limits = {name: [0, 0] for name in source_prefixes}
    source_settings = [
        {"color": channel_colors[name], "contrastLimits": contrast_limits[name], "visible": True}
        for name in source_prefixes
    ]

    htm.add_plate_grid_view(
        ds_folder, view_name="default", source_prefixes=source_prefixes, source_types=source_types,
        source_settings=source_settings,
        source_name_to_site_name=to_site_name,
        site_name_to_well_name=to_well_name,
        site_table=site_table, well_table=well_table,
        sites_visible=False, menu_name="bookmark"
    )


def convert_to_mobie_nested(plate_config):
    in_root = os.path.join(INPUT_ROOT, plate_config.folder)

    if not mobie.metadata.project_exists(OUTPUT_ROOT):
        mobie.metadata.create_project_metadata(OUTPUT_ROOT)

    pos_to_pattern = {}
    ds_folder = os.path.join(
        OUTPUT_ROOT,
        os.path.basename(plate_config.folder).lower()
    )
    input_folders = glob(os.path.join(in_root, "*"))
    for in_folder in input_folders:
        if not os.path.isdir(in_folder):
            assert False, in_folder
        pos_to_pattern = convert_images(in_folder, ds_folder, pos_to_pattern, plate_config.channel_order)

    site_table, well_table = get_tables(pos_to_pattern, os.path.join(ds_folder, "tables"), plate_config.to_site_name)
    add_grid_view(ds_folder, site_table, well_table, plate_config.channel_order, plate_config.channel_colors)

    mobie.metadata.set_is2d(ds_folder, True)


def convert_to_mobie(plate_config):
    in_folder = os.path.join(INPUT_ROOT, plate_config.folder)

    if not mobie.metadata.project_exists(OUTPUT_ROOT):
        mobie.metadata.create_project_metadata(OUTPUT_ROOT)

    ds_folder = os.path.join(
        OUTPUT_ROOT, os.path.basename(plate_config.folder).lower()
    )
    pos_to_pattern = convert_images(in_folder, ds_folder,
                                    pos_to_pattern={},
                                    channel_order=plate_config.channel_order)

    site_table, well_table = get_tables(pos_to_pattern, os.path.join(ds_folder, "tables"), plate_config.to_site_name)
    add_grid_view(ds_folder, site_table, well_table,
                  plate_config.channel_order, plate_config.channel_colors,
                  plate_config.to_site_name)

    mobie.metadata.set_is2d(ds_folder, True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")  # e.g. "./plate_configs/mix_wt_alpha_control.json"
    args = parser.parse_args()
    plate_config = read_plate_config(args.config_file)
    if plate_config.nested:
        convert_to_mobie_nested(plate_config)
    else:
        convert_to_mobie(plate_config)


if __name__ == "__main__":
    main()
