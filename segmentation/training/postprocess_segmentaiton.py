import os
from glob import glob

import numpy as np
from skimage.measure import regionprops
from elf.io import open_file


def postprocess_segmentation(image_path, seg_path, output_path):
    with open_file(image_path, "r") as f:
        image = f["s0"][:]
    with open_file(seg_path, "r") as f:
        seg = f["s0"][:]

    # filter bg segments
    props = regionprops(seg, image)
    seg_ids = np.array([prop.label for prop in props])
    intensities = np.array([prop.mean_intensity for prop in props])

    bg_intensity = np.mean(image[seg == 0])
    fg_intensity = np.mean(image[seg != 0])
    print(bg_intensity)
    print(fg_intensity)

    min_cell_intensity = ""
    discard_ids = seg_ids[intensities < min_cell_intensity]
    seg[np.isin(seg, discard_ids)] = 0

    return
    with open_file(output_path, "w") as f:
        f.create_dataset("image", data=image, compression="gzip")
        f.create_dataset("labels", data=seg, compression="gzip")


def main():
    data_folder = ""
    image_paths = glob(os.path.join(data_folder, "*serum*"))
    image_paths.sort()
    seg_paths = glob(os.path.join(data_folder, "*cell-segmentation*"))
    seg_paths.sort()

    output_folder = ""
    os.makedirs(output_folder, exist_ok=True)
    for sample, (im_path, seg_path) in enumerate(zip(image_paths, seg_paths)):
        out_path = os.path.join(output_folder, f"S{sample}.h5")
        postprocess_segmentation(im_path, seg_path, out_path)
        return


if __name__ == "__main__":
    pass
