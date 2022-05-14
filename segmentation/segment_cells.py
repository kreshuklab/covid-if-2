import os
from glob import glob

import bioimageio.core
import h5py
from bioimageio.core.prediction import predict_with_tiling
from tqdm import tqdm
from xarray import DataArray


def predict_image(pp, in_path, out_path):
    fg_key = "prediction/foreground"
    bd_key = "prediction/boundaries"
    tiling = {
        "tile": {"x": 1024, "y": 1024}, "halo": {"x": 64, "y": 64}
    }
    with h5py.File(in_path, "a") as f:
        if bd_key in f:
            return
        image = f["serum"][:]
        image = DataArray(image[None, None], dims=tuple("bcyx"))
        pred = predict_with_tiling(pp, image, tiling=tiling)[0].values
        with h5py.File(out_path, "w") as f:
            f.create_dataset(fg_key, data=pred[0], compression="gzip")
            f.create_dataset(bd_key, data=pred[1], compression="gzip")


def run_prediction(input_folder, output_folder):
    inputs = glob(os.path.join(input_folder, "*.h5"))

    doi = "10.5281/zenodo.5847355"
    model = bioimageio.core.load_resource_description(doi)

    with bioimageio.core.create_prediction_pipeline(model) as pp:
        for in_path in tqdm(inputs, desc="Run model prediction"):
            out_path = os.path.join(output_folder, os.path.basename(in_path))
            predict_image(pp, in_path, out_path)


# TODO
def segment_image(path):
    pass


def run_watershed(folder):
    files = glob(os.path.join(folder, "*.h5"))
    for path in tqdm(files, desc="Run watershed"):
        segment_image(path)


def run_cell_segmentation(input_folder, output_folder):
    run_prediction(input_folder, output_folder)
    run_watershed(output_folder)


def main():
    input_folder = "/home/pape/Work/data/covid-if2/test-data"
    output_folder = "/home/pape/Work/data/covid-if2/test-data-processed"
    run_cell_segmentation(input_folder, output_folder)


if __name__ == "__main__":
    main()
