import os
from glob import glob

import h5py
import elf.htm as htm
from tqdm import tqdm


def load_channel(files, channel):
    data = {}
    for sample, ff in tqdm(enumerate(files, 1), total=len(files), desc=f"Load {channel}"):
        with h5py.File(ff, "r") as f:
            im = f[channel][:]
        data[sample] = im
    return data


def load_positions(files):
    positions = {}
    for sample, ff in enumerate(files, 1):
        with h5py.File(ff, "r") as f:
            y, x = f.attrs["y"], f.attrs["x"]
            positions[sample] = (y, x)
    return positions


# TODO group samples by position to recover the well labels
def check_segmentation_results():
    folder = "/home/pape/Work/data/covid-if2/processed-v1"
    files = glob(os.path.join(folder, "*.h5"))
    files.sort()

    positions = load_positions(files)

    # image_channels = ["serum", "marker", "nuclei"]
    image_channels = ["serum"]
    image_data = {channel: load_channel(files, channel) for channel in image_channels}
    image_settings = {
        "serum": {"colormap": "green"},
        "marker": {"colormap": "red"},
        "nuclei": {"colormap": "blue"},
    }

    # label_channels = ["segmentation/cells", "segmentation/nuclei"]
    label_channels = ["segmentation/cells"]
    label_data = {channel: load_channel(files, channel) for channel in label_channels}

    htm.view_positional_images(image_data, positions, label_data, image_settings)


if __name__ == "__main__":
    check_segmentation_results()
