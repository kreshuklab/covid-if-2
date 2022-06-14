import argparse
import os
from glob import glob

import h5py
import elf.htm as htm
from tqdm import tqdm


ALL_CHANNELS = ["serum", "marker", "nuclei"]
ALL_LABELS = ["segmentation/cells", "segmentation/nuclei"]


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


def check_positional(folder, seg_folder, image_channels, label_channels):
    files = glob(os.path.join(folder, "*.h5"))
    files.sort()

    positions = load_positions(files)

    image_data = {channel: load_channel(files, channel) for channel in image_channels}
    image_settings = {
        "serum": {"colormap": "green", "blending": "additive"},
        "marker": {"colormap": "red", "blending": "additive"},
        "nuclei": {"colormap": "blue", "blending": "additive"},
    }

    files = glob(os.path.join(seg_folder, "*.h5"))
    files.sort()
    if label_channels:
        label_data = {channel: load_channel(files, channel) for channel in label_channels}
    else:
        label_data = None

    htm.view_positional_images(image_data, positions, label_data, image_settings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--import_folder", required=True)
    parser.add_argument("-c", "--channels", type=str, nargs="+", default=ALL_CHANNELS)
    # parser.add_argument("-l", "--labels", type=str, nargs="+", default=ALL_LABELS)
    parser.add_argument("-l", "--labels", type=str, nargs="+", default=[])
    args = parser.parse_args()
    folder = args.import_folder
    seg_folder = folder.replace("imported", "processed")
    check_positional(folder, seg_folder, args.channels, args.labels)


if __name__ == "__main__":
    main()
