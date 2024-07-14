import os
from glob import glob

import bioimageio.core
import numpy as np
import pandas as pd
import vigra
import zarr

from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.transform import resize
from xarray import DataArray
from tqdm import tqdm


def segment_cells(model, image, seeds):
    tiling = {"tile": {"x": 1536, "y": 1536}, "halo": {"x": 256, "y": 256}}
    image = DataArray(image[None, None], dims=tuple("bcyx"))
    pred = bioimageio.core.predict_with_tiling(model, image, tiling=tiling)[0].values[0]
    assert pred.shape[0] == 2
    fg, hmap = pred[0], pred[1]

    seed_ids = np.unique(seeds)
    shrink_radius = 7
    shrink_mask = distance_transform_edt(seeds != 0)
    seeds_shrunk = seeds.copy()
    seeds_shrunk[shrink_mask < shrink_radius] = 0
    ids_after_shrinking = np.unique(seeds_shrunk)
    missing_ids = np.setdiff1d(seed_ids, ids_after_shrinking)
    if len(missing_ids) > 0:
        reinsert_mask = np.isin(seeds, missing_ids)
        seeds_shrunk[reinsert_mask] = seeds[reinsert_mask]

    bg_seed_id = int(np.max(seeds_shrunk) + 1)

    bg_seed_mask = (fg - hmap) < 0.5
    seeds_shrunk[bg_seed_mask] = bg_seed_id

    seg = watershed(hmap, seeds_shrunk)
    seg[seg == bg_seed_id] = 0

    return seg


def update_table(seg, out_table_path):
    ndim = 2
    resolution = [1.0, 1.0]

    centers = vigra.filters.eccentricityCenters(seg.astype("uint32"))
    props = regionprops(seg)
    tab = np.array([
        [p.label]
        + [ce / res for ce, res in zip(centers[p.label], resolution)]
        + [float(bb) / res for bb, res in zip(p.bbox[:ndim], resolution)]
        + [float(bb) / res for bb, res in zip(p.bbox[ndim:], resolution)]
        + [p.area]
        for p in props
    ])

    col_names = ["label_id", "anchor_y", "anchor_x",
                 "bb_min_y", "bb_min_x", "bb_max_y", "bb_max_x", "n_pixels"]
    assert tab.shape[1] == len(col_names), f"{tab.shape}, {len(col_names)}"
    tab = pd.DataFrame(tab, columns=col_names)
    tab.to_csv(out_table_path, sep="\t", index=False, na_rep="nan")


def write_segmentation(seg, out_path):
    with zarr.open(out_path, "a") as f:
        ds = f["s0"]
        ds[:] = seg
        for scale in range(1, len(f)):
            ds = f[f"s{scale}"]
            seg_scale = resize(
                seg, ds.shape, order=0, preserve_range=True, anti_aliasing=False
            ).astype(seg.dtype)
            ds[:] = seg_scale


def update_segmentation(model, image_path, seed_path, out_seg_path, out_table_path):
    with zarr.open(image_path, "r") as f:
        image = f["s0"][:]
    with zarr.open(seed_path, "r") as f:
        seeds = f["s0"][:]
    seg = segment_cells(model, image, seeds)
    assert seg.shape == image.shape

    write_segmentation(seg, out_seg_path)
    update_table(seg, out_table_path)


def update_cell_segmentations():
    doi = "/scratch/pape/covid-if-2/networks/segmentation/instance-pseudo-labels/instance-pseudo-labels.zip"

    ds_folder = "/scratch/pape/covid-if-2/data/markers_new"
    image_paths = glob(os.path.join(ds_folder, "images", "ome-zarr", "*serum*"))
    image_paths.sort()
    seed_paths = glob(os.path.join(ds_folder, "images", "ome-zarr", "*nucleus-segmentation*"))
    seed_paths.sort()
    output_paths = glob(os.path.join(ds_folder, "images", "ome-zarr", "*cell-segmentation*"))
    output_paths.sort()
    table_paths = glob(os.path.join(ds_folder, "tables", "*cell-segmentation*"))
    table_paths.sort()
    assert len(image_paths) == len(seed_paths) == len(output_paths) == len(table_paths)
    print("Updating", len(image_paths), "segmentations")

    model = bioimageio.core.load_resource_description(doi)
    with bioimageio.core.create_prediction_pipeline(model) as pp:
        for im_path, seed_path, out_path, table_folder in tqdm(
            zip(image_paths, seed_paths, output_paths, table_paths), total=len(image_paths)
        ):
            table_path = os.path.join(table_folder, "default.tsv")
            update_segmentation(pp, im_path, seed_path, out_path, table_path)


if __name__ == "__main__":
    update_cell_segmentations()
