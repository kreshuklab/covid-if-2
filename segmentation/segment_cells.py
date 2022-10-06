import os
from glob import glob

import bioimageio.core
import h5py
import numpy as np

from skimage.segmentation import watershed
from scipy.ndimage.morphology import binary_dilation
from tqdm import tqdm
from xarray import DataArray


def predict_image(pp, in_path, out_path):
    fg_key = "prediction/foreground"
    bd_key = "prediction/boundaries"
    with h5py.File(in_path, "a") as f:
        if bd_key in f:
            return

    tiling = {
        "tile": {"x": 1536, "y": 1536}, "halo": {"x": 128, "y": 128}
    }

    with h5py.File(in_path, "r") as f:
        image = f["serum"][:]

    image = DataArray(image[None, None], dims=tuple("bcyx"))
    pred = bioimageio.core.predict_with_tiling(pp, image, tiling=tiling)[0].values[0]

    with h5py.File(out_path, "a") as f:
        f.create_dataset(fg_key, data=pred[0], compression="gzip")
        f.create_dataset(bd_key, data=pred[1], compression="gzip")


def run_prediction(input_folder, output_folder):
    inputs = glob(os.path.join(input_folder, "*.h5"))
    os.makedirs(output_folder, exist_ok=True)

    doi = "10.5281/zenodo.5847355"
    model = bioimageio.core.load_resource_description(doi)

    with bioimageio.core.create_prediction_pipeline(model) as pp:
        for in_path in tqdm(inputs, desc="Run model prediction"):
            out_path = os.path.join(
                output_folder, os.path.basename(in_path)
            )
            predict_image(pp, in_path, out_path)


def segment_image(path):
    fg_key = "prediction/foreground"
    bd_key = "prediction/boundaries"
    seed_key = "segmentation/nuclei"
    with h5py.File(path, "r") as f:
        foreground = f[fg_key][:]
        hmap = f[bd_key][:]
        seeds = f[seed_key][:]

    # TODO determine from data
    min_seed_size = 250
    seed_ids, seed_sizes = np.unique(seeds, return_counts=True)
    remove_seeds = seed_ids[seed_sizes < min_seed_size]
    seeds[np.isin(seeds, remove_seeds)] = 0

    threshold = 0.5
    seed_dilation = 5
    mask = foreground > threshold
    seed_mask = binary_dilation(seeds, iterations=seed_dilation)
    mask = np.logical_or(mask, seed_mask)
    cells = watershed(hmap, markers=seeds, mask=mask)

    seg_key = "segmentation/cells"
    with h5py.File(path, "a") as f:
        f.create_dataset(seg_key, data=cells, compression="gzip")


def run_watershed(folder):
    files = glob(os.path.join(folder, "*.h5"))
    for path in tqdm(files, desc="Run watershed"):
        segment_image(path)


# TODO switch to model with percentile normalization
# compute it globally per image or well (check out Fynn's example notebook on this)
# idea for model retraining:
# per cell intensity augmentation that lower the intensity for explicit (smoothed) cell masks in the training images.
def run_cell_segmentation(input_folder, output_folder):
    run_prediction(input_folder, output_folder)
    run_watershed(output_folder)


def main():
    folder = "/home/pape/Work/data/covid-if2/processed-v1"
    run_cell_segmentation(folder)


if __name__ == "__main__":
    main()
