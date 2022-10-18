import os
from glob import glob

import h5py
import napari


def check_sample(in_path, out_path):
    inputs = {}
    with h5py.File(in_path, "r") as f:
        for name, ds in f.items():
            inputs[name] = ds[:]

    segmentations = {}
    predictions = {}
    with h5py.File(out_path, "r") as f:
        if "prediction" in f:
            for name, ds in f["prediction"].items():
                predictions[name] = ds[:]
        if "segmentation" in f:
            for name, ds in f["segmentation"].items():
                segmentations[name] = ds[:]

    v = napari.Viewer()
    v.title = os.path.basename(in_path)
    for name, data in inputs.items():
        v.add_image(data, name=f"input/{name}")
    for name, data in predictions.items():
        v.add_image(data, name=f"prediction/{name}")
    for name, data in segmentations.items():
        v.add_labels(data, name=f"segmentation/{name}")
    napari.run()


def check_segmentation(input_folder, processed_folder):
    inputs = glob(os.path.join(input_folder, "*.h5"))
    for inp in inputs:
        outp = os.path.join(processed_folder, os.path.basename(inp))
        check_sample(inp, outp)


def main():
    input_folder = "/home/pape/Work/data/covid-if2/test-data"
    processed_folder = "/home/pape/Work/data/covid-if2/test-data-processed"
    check_segmentation(input_folder, processed_folder)


if __name__ == "__main__":
    main()
