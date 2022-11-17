import argparse
import os
from glob import glob

import mobie
import z5py
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from tqdm import tqdm

from plate_utils import read_plate_config

OUTPUT_ROOT = "/scratch/pape/covid-if-2/data"


def segment_nuclei(model, path):
    with z5py.File(path, "r") as f:
        image = f["s0"][:]
    input_ = normalize(image, 1.0, 99.8)
    scale = 1  # maybe try other scales
    nuclei, _ = model.predict_instances(input_, scale=scale)
    return nuclei


def run_nucleus_segmentation(ds_name):
    ds_folder = os.path.join(OUTPUT_ROOT, ds_name)
    assert os.path.exists(ds_folder), ds_folder
    sources = mobie.metadata.read_dataset_metadata(ds_folder)["sources"]

    image_folder = os.path.join(ds_folder, "images", "ome-zarr")
    inputs = glob(os.path.join(image_folder, "nuclei*"))
    model = None

    resolution = [1.0, 1.0]
    chunks = (1024, 1024)
    scale_factors = [[2, 2], [2, 2]]

    for in_path in tqdm(inputs, desc="Run stardist prediction"):
        fname = os.path.basename(in_path)[:-9]
        pos = fname.split("_")[1]
        source_name = f"nucleus-segmentation_{pos}"
        if source_name in sources:
            continue

        if model is None:
            model = StarDist2D.from_pretrained("2D_versatile_fluo")
        seg = segment_nuclei(model, in_path)
        tmp_folder = f"tmps/{source_name}"
        mobie.add_segmentation(seg, None, OUTPUT_ROOT, ds_name, source_name,
                               resolution=resolution, view={}, tmp_folder=tmp_folder,
                               scale_factors=scale_factors, chunks=chunks,
                               unit="pixel", file_format="ome.zarr", add_default_table=False,
                               max_jobs=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")  # e.g. "./plate_configs/mix_wt_alpha_control.json"
    args = parser.parse_args()
    plate_config = read_plate_config(args.config_file)
    folder_name = os.path.basename(plate_config.folder).lower()
    run_nucleus_segmentation(folder_name)


if __name__ == "__main__":
    main()
