import argparse
import os
from glob import glob

import bioimageio.core
import mobie
import mobie.htm as htm
import numpy as np
import z5py

from skimage.segmentation import watershed
from scipy.ndimage.morphology import binary_dilation
from tqdm import tqdm
from xarray import DataArray

from plate_utils import to_well_name, to_position, read_plate_config, OUTPUT_ROOT


def segment_cells(model, serum_path, nucleus_path):
    with z5py.File(serum_path, "r") as f:
        image = f["s0"][:]

    tiling = {"tile": {"x": 1536, "y": 1536}, "halo": {"x": 128, "y": 128}}
    image = DataArray(image[None, None], dims=tuple("bcyx"))
    pred = bioimageio.core.predict_with_tiling(model, image, tiling=tiling)[0].values[0]
    assert pred.shape[0] == 2
    foreground, hmap = pred[0], pred[1]

    with z5py.File(nucleus_path, "r") as f:
        seeds = f["s0"][:]

    threshold = 0.5
    seed_dilation = 6
    mask = foreground > threshold
    seed_mask = binary_dilation(seeds, iterations=seed_dilation)
    mask = np.logical_or(mask, seed_mask)
    cells = watershed(hmap, markers=seeds, mask=mask)

    return cells


def run_cell_segmentation(ds_name):
    ds_folder = os.path.join(OUTPUT_ROOT, ds_name)
    assert os.path.exists(ds_folder), ds_folder
    sources = mobie.metadata.read_dataset_metadata(ds_folder)["sources"]

    image_folder = os.path.join(ds_folder, "images", "ome-zarr")
    serum_paths = glob(os.path.join(image_folder, "serum*"))
    serum_paths.sort()
    nucleus_paths = glob(os.path.join(image_folder, "nucleus-segmentation*"))
    nucleus_paths.sort()
    assert len(serum_paths) == len(nucleus_paths)

    resolution = [1.0, 1.0]
    chunks = (1024, 1024)
    scale_factors = [[2, 2], [2, 2]]

    # doi = "10.5281/zenodo.5847355"
    # the updated model:
    doi = "/scratch/pape/covid-if-2/networks/segmentation/instance-pseudo-labels/instance-pseudo-labels.zip"
    model = None

    for serum, nuc in tqdm(zip(serum_paths, nucleus_paths), total=len(serum_paths), desc="Run cell segmentation"):
        fname = os.path.basename(serum)[:-9]
        pos = fname.split("_")[1]
        source_name = f"cell-segmentation_{pos}"
        if source_name in sources:
            continue

        if model is None:
            model = bioimageio.core.load_resource_description(doi)
            pp = bioimageio.core.create_prediction_pipeline(model)
        seg = segment_cells(pp, serum, nuc)

        tmp_folder = f"tmps/{source_name}"
        mobie.add_segmentation(seg, None, OUTPUT_ROOT, ds_name, source_name,
                               resolution=resolution, view={}, tmp_folder=tmp_folder,
                               scale_factors=scale_factors, chunks=chunks,
                               unit="pixel", file_format="ome.zarr", max_jobs=8)


def add_grid_view(ds_name, channel_order, channel_colors, to_site_name):
    ds_folder = os.path.join(OUTPUT_ROOT, ds_name)

    source_prefixes = list(channel_order.values())
    seg_prefixes = ["nucleus-segmentation", "cell-segmentation"]

    contrast_limits = {
        name: htm.compute_contrast_limits(
            name, ds_folder, lower_percentile=4, upper_percentile=96, n_threads=16
        )
        for name in source_prefixes
    }

    source_settings = [
        {"color": channel_colors[name], "contrastLimits": contrast_limits[name], "visible": True}
        for name in source_prefixes
    ] + [
        {"lut": "glasbey", "visible": False, "opacity": 0.5},
        {"lut": "glasbey", "visible": False, "opacity": 0.5, "showTable": False},
    ]

    source_prefixes += seg_prefixes
    source_types = len(channel_order) * ["image"] + 2 * ["segmentation"]
    htm.add_plate_grid_view(
        ds_folder, view_name="segmentations", source_prefixes=source_prefixes, source_types=source_types,
        source_settings=source_settings, source_name_to_site_name=to_site_name,
        site_name_to_well_name=to_well_name, well_to_position=to_position,
        site_table="sites", well_table="wells",
        sites_visible=False, menu_name="bookmark"
    )


# ideas to improve segmentation
# - use pseudo-training model
# - subtract boundaries from fg before computing mask
# - try second round of pseudo-labeling
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")  # e.g. "./plate_configs/mix_wt_alpha_control.json"
    args = parser.parse_args()
    plate_config = read_plate_config(args.config_file)
    folder_name = os.path.basename(plate_config.folder).lower()
    run_cell_segmentation(folder_name)
    add_grid_view(folder_name, plate_config.channel_order, plate_config.channel_colors, plate_config.to_site_name)


if __name__ == "__main__":
    main()
