import os
from glob import glob

import h5py
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from tqdm import tqdm


def segment_nuclei(model, in_path, out_path, key="nuclei"):
    with h5py.File(in_path, "r") as f:
        image = f[key][:]
    input_ = normalize(image, 1.0, 99.8)
    scale = 1  # maybe try other scales
    nuclei, _ = model.predict_instances(input_, scale=scale)
    out_key = "segmentation/nuclei"
    with h5py.File(out_path, "a") as f:
        f.create_dataset(out_key, data=nuclei, compression="gzip")


def run_nucleus_segmentation(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    inputs = glob(os.path.join(input_folder, "*.h5"))
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    for in_path in tqdm(inputs, desc="Run stardist prediction"):
        out_path = os.path.join(output_folder, os.path.basename(in_path))
        segment_nuclei(model, in_path, out_path)


def main():
    input_folder = "/home/pape/Work/data/covid-if2/test-data"
    output_folder = "/home/pape/Work/data/covid-if2/test-data-processed"
    run_nucleus_segmentation(input_folder, output_folder)


if __name__ == "__main__":
    main()
