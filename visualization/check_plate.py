import os
from glob import glob

import h5py
import elf.htm as htm
from tqdm import tqdm


def sample_to_well(sample):
    if 0 <= sample < 8:
        well = "B2"
    elif 8 <= sample < 16:
        well = "B3"
    elif 16 <= sample < 24:
        well = "B4"
    elif 24 <= sample < 32:
        well = "B5"
    elif 32 <= sample < 40:
        well = "B6"
    elif 40 <= sample < 48:
        well = "B7"
    # skip the C-well, can't deal with incomplete wells for now
    else:
        well = None
    return well


def load_channel(files, channel):
    data = {}
    for sample, ff in tqdm(enumerate(files), total=len(files), desc=f"Load {channel}"):
        well = sample_to_well(sample)
        if well is None:
            continue
        well_data = data.get(well, [])
        with h5py.File(ff, "r") as f:
            im = f[channel][:]
        well_data.append(im)
        data[well] = well_data
    return data


# group samples by well
def check_segmentation_results():
    image_folder = "/home/pape/Work/data/covid-if2/imported/test-data"
    files = glob(os.path.join(image_folder, "*.h5"))
    files.sort()

    # image_channels = ["serum", "marker", "nuclei"]
    image_channels = ["serum"]
    image_data = {channel: load_channel(files, channel) for channel in image_channels}
    image_settings = {
        "serum": {"colormap": "green", "blending": "additive"},
        "marker": {"colormap": "red", "blending": "additive"},
        "nuclei": {"colormap": "blue", "blending": "additive"},
    }

    res_folder = "/home/pape/Work/data/covid-if2/processed/test-data"
    files = glob(os.path.join(res_folder, "*.h5"))
    files.sort()
    # label_channels = ["segmentation/cells", "segmentation/nuclei"]
    label_channels = ["segmentation/cells"]
    label_data = {channel: load_channel(files, channel) for channel in label_channels}

    htm.view_plate(image_data, label_data, image_settings, well_shape=(4, 2), site_spacing=8, well_spacing=64)


if __name__ == "__main__":
    check_segmentation_results()
