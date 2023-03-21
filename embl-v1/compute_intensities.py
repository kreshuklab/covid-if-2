import argparse
import os
from concurrent import futures
from functools import partial

import mobie
import numpy as np
import pandas as pd
from elf.io import open_file
from tqdm import tqdm

from plate_utils import read_plate_config, write_plate_config, OUTPUT_ROOT


def median_absolute_deviation(x):
    median = np.median(x)
    mad = np.abs(x - median).sum() / float(x.size)
    return mad


def saturation_ratio(x, saturation_threshold=16384):
    return float((x >= saturation_threshold).sum()) / x.sum()


# robust statistics
STATS = {
    "median": np.median,
    "q05": partial(np.percentile, q=5),
    "q95": partial(np.percentile, q=95),
    "mad": median_absolute_deviation,
    "mean": np.mean,
    "saturation_ratio": saturation_ratio,
}


def compute_stats(ds_folder, position, seg_name, pos_sources, sources):
    seg_source = [source for source in pos_sources if seg_name in source]
    assert len(seg_source) == 1
    seg_source = seg_source[0]

    seg_path = os.path.join(
        ds_folder, sources[seg_source]["segmentation"]["imageData"]["ome.zarr"]["relativePath"]
    )
    assert os.path.exists(seg_path), seg_path
    with open_file(seg_path, "r") as f:
        seg = f["s0"][:]

    channels = {}
    for source in pos_sources:
        if "segmentation" in source:
            continue
        image_path = os.path.join(
            ds_folder, sources[source]["image"]["imageData"]["ome.zarr"]["relativePath"]
        )
        channel_name = source.split("_")[0]
        assert os.path.exists(image_path)
        with open_file(image_path, "r") as f:
            channels[channel_name] = f["s0"][:]

    rows = {"label_id": [], "size": []}
    rows.update({f"{channel_name}_{stat}": [] for channel_name in channels.keys() for stat in STATS})
    seg_ids = np.unique(seg)
    for seg_id in seg_ids:
        mask = seg == seg_id
        rows["label_id"].append(seg_id)
        for channel_name, channel in channels.items():
            x = channel[mask]
            for stat, func in STATS.items():
                rows[f"{channel_name}_{stat}"].append(func(x))

        # statistics independent of intensity
        rows["size"].append(mask.sum())

    table = pd.DataFrame.from_dict(rows)
    return table


def stats_impl(position, grid_layout, ds_folder, sources):
    reference_segmentation = "cell-segmentation"
    segmentation_names = ["nucleus-segmentation", "cell-segmentation"]

    pos_sources = grid_layout[position]
    ref_source = [source for source in pos_sources if reference_segmentation in source]
    assert len(ref_source) == 1
    ref_source = ref_source[0]

    table_folder = os.path.join(
        ds_folder, sources[ref_source]["segmentation"]["tableData"]["tsv"]["relativePath"]
    )
    assert os.path.exists(table_folder)

    bg_stats = None
    for seg_name in segmentation_names:
        table = compute_stats(ds_folder, position, seg_name, pos_sources, sources)

        # pop the first row (background label), keep track of it if this is the ref seg
        table, this_bg_stats = table.drop(table.head(1).index), table.head(1)
        assert this_bg_stats.label_id.all() == 0
        this_bg_stats = this_bg_stats.drop(columns="label_id")
        if seg_name == reference_segmentation:
            bg_stats = this_bg_stats

        # save the table
        table_path = os.path.join(table_folder, f"statistics_{seg_name}.tsv")
        table.to_csv(table_path, sep="\t", index=False)

    return bg_stats


def compute_segmentation_statistics(metadata, ds_folder, site_table, channel_order):
    site_stat_table = os.path.join(ds_folder, "tables", "sites", "bg_stats.tsv")

    # get the grid layout to find all the positions for this source
    view = metadata["views"]["segmentations"]
    displays = view["sourceDisplays"]
    grid_layout = None
    for display in displays:
        type_, values = next(iter(display.items()))
        if type_ == "regionDisplay" and values["name"] == "sites":
            grid_layout = values["sources"]
    assert grid_layout is not None

    sources = metadata["sources"]

    n_workers = 32
    positions = site_table["region_id"].values
    compute_stats = partial(stats_impl, grid_layout=grid_layout, ds_folder=ds_folder, sources=sources)

    # for debugging
    # bg_stats = compute_stats(positions[0])

    # compute statistics for all images in parallel
    with futures.ProcessPoolExecutor(n_workers) as pp:
        bg_stats = list(tqdm(
            pp.map(compute_stats, positions), desc="Compute per cell statistics", total=len(positions)
        ))

    bg_stats = pd.concat(bg_stats)
    region_ids = site_table["region_id"]
    assert len(region_ids) == len(bg_stats)
    bg_stats.insert(loc=0, column="region_id", value=region_ids.values)
    bg_stats.to_csv(site_stat_table, sep="\t", index=False)


def compute_statistics(ds_folder, channel_order):
    metadata = mobie.metadata.read_dataset_metadata(ds_folder)

    site_table_folder = os.path.join(ds_folder, "tables", "sites")
    site_table = pd.read_csv(os.path.join(site_table_folder, "default.tsv"), sep="\t")

    # compute the per cell intensities
    compute_segmentation_statistics(metadata, ds_folder, site_table, channel_order)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")  # e.g. "./plate_configs/mix_wt_alpha_control.json"
    args = parser.parse_args()
    plate_config = read_plate_config(args.config_file)

    if plate_config.processed["compute_intensities"]:
        return

    folder_name = os.path.basename(plate_config.folder).lower()
    ds_folder = os.path.join(OUTPUT_ROOT, folder_name)
    compute_statistics(ds_folder, plate_config.channel_order)

    plate_config.processed["compute_intensities"] = True
    write_plate_config(args.config_file, plate_config)


if __name__ == "__main__":
    main()
