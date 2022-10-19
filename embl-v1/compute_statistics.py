import os
from concurrent import futures
from functools import partial

import numpy as np
import pandas as pd
from elf.io import open_file
from tqdm import tqdm

from plate_utils import to_well_name, CHANNEL_ORDER

OUTPUT_ROOT = "/scratch/pape/covid-if-2/data"


def median_absolute_deviation(x):
    median = np.median(x)
    mad = np.abs(x - median).sum() / float(x.size)
    return mad


# robust statistics
STATS = {
    "median": np.median,
    "q05": partial(np.percentile, q=5),
    "q95": partial(np.percentile, q=95),
    "mad": median_absolute_deviation,
}


def compute_stats(ds_folder, position, seg_name, channel_names):
    pos_name = position.capitalize()
    seg_path = os.path.join(ds_folder, "images", "ome-zarr", f"{seg_name}_{pos_name}.ome.zarr")
    with open_file(seg_path, "r") as f:
        seg = f["s0"][:]

    channels = {}
    for channel_name in channel_names:
        image_path = os.path.join(ds_folder, "images", "ome-zarr", f"{channel_name}_{pos_name}.ome.zarr")
        with open_file(image_path, "r") as f:
            channels[channel_name] = f["s0"][:]

    rows = {"label_id": []}
    rows.update({f"{channel_name}_{stat}": [] for channel_name in channel_names for stat in STATS})
    seg_ids = np.unique(seg)
    for seg_id in seg_ids:
        mask = seg == seg_id
        rows["label_id"].append(seg_id)
        for channel_name, channel in channels.items():
            x = channel[mask]
            for stat, func in STATS.items():
                rows[f"{channel_name}_{stat}"].append(func(x))

    table = pd.DataFrame.from_dict(rows)
    return table


def stats_impl(position, ds_folder):
    pos_name = position.capitalize()
    channel_names = list(CHANNEL_ORDER.values())
    reference_segmentation = "cell-segmentation"
    segmentation_names = ["nucleus-segmentation", "cell-segmentation"]

    bg_stats = None
    for seg_name in segmentation_names:
        table = compute_stats(ds_folder, position, seg_name, channel_names)

        # pop the first row (background label), keep track of it if this is the ref seg
        table, this_bg_stats = table.drop(table.head(1).index), table.head(1)
        assert this_bg_stats.label_id.all() == 0
        this_bg_stats = this_bg_stats.drop(columns="label_id")
        if seg_name == reference_segmentation:
            bg_stats = this_bg_stats

        # save the table
        table_path = os.path.join(
            ds_folder, "tables", f"{reference_segmentation}_{pos_name}", f"statistics_{seg_name}.tsv"
        )
        table.to_csv(table_path, sep="\t", index=False)

    return bg_stats


def compute_segmentation_statistics(ds_folder, site_table, force):
    site_stat_table = os.path.join(ds_folder, "tables", "sites", "bg_stats.tsv")
    if os.path.exists(site_stat_table) and not force:
        return

    n_workers = 32
    positions = site_table["position"].values
    compute_stats = partial(stats_impl, ds_folder=ds_folder)

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


def compute_site_statistics(ds_folder, site_table, force):
    site_stat_table = os.path.join(ds_folder, "tables", "sites", "statistics.tsv")
    if os.path.exists(site_stat_table) and not force:
        return

    channel_names = list(CHANNEL_ORDER.values())
    region_ids = site_table["region_id"].values
    positions = site_table["position"].values

    rows = {"region_id": region_ids}
    rows.update({f"{channel_name}_{stat}": [] for channel_name in channel_names for stat in STATS})

    for position in tqdm(positions):
        for channel_name in channel_names:
            image_path = os.path.join(
                ds_folder, "images", "ome-zarr", f"{channel_name}_{position.capitalize()}.ome.zarr"
            )
            with open_file(image_path, "r") as f:
                data = f["s0"][:]
            for stat, func in STATS.items():
                rows[f"{channel_name}_{stat}"].append(func(data))

    table = pd.DataFrame.from_dict(rows)
    table.to_csv(site_stat_table, sep="\t", index=False)


# currently we don't need the well level stats
def compute_well_statistics(ds_folder, site_table, force):
    pass
    # well_stat_table = os.path.join(ds_folder, "tables", "sites", "bg_stats.tsv")


def compute_statistics(ds_folder, force=False):
    well_table_folder = os.path.join(ds_folder, "tables", "wells")
    well_table = pd.read_csv(os.path.join(well_table_folder, "default.tsv"), sep="\t")

    site_table_folder = os.path.join(ds_folder, "tables", "sites")
    site_table = pd.read_csv(os.path.join(site_table_folder, "default.tsv"), sep="\t")

    compute_segmentation_statistics(ds_folder, site_table, force)

    # implement the agglomerations over the cell statistic per site and well here,
    # (will be helpful in general esp. for QC, but we don't need it right now)
    compute_site_statistics(ds_folder, site_table, force)
    compute_well_statistics(ds_folder, well_table, force)


def main():
    ds_name = "markers_new"
    ds_folder = os.path.join(OUTPUT_ROOT, ds_name)
    compute_statistics(ds_folder, force=True)


if __name__ == "__main__":
    main()
