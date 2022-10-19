import os
from glob import glob

import napari
from elf.io import open_file


def postprocess_segmentation(image_path, seg_path, output_path):
    with open_file(image_path, "r") as f:
        image = f["s0"][:]
    with open_file(seg_path, "r") as f:
        seg = f["s0"][:]

    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(seg, name="segmentation")

    @v.bind_key("s")
    def save_seg(v):
        print("updating the segmentation")
        nonlocal seg
        seg = v.layers["segmentation"].data

    napari.run()

    with open_file(output_path, "w") as f:
        f.create_dataset("image", data=image, compression="gzip")
        f.create_dataset("labels", data=seg, compression="gzip")


def main():
    data_folder = "/scratch/pape/covid-if-2/data/markers_new/images/ome-zarr"
    image_paths = glob(os.path.join(data_folder, "*serum*"))
    image_paths.sort()
    seg_paths = glob(os.path.join(data_folder, "*cell-segmentation*"))
    seg_paths.sort()

    output_folder = "/scratch/pape/covid-if-2/training_data/segmentation/pseudo-labels-v1"
    os.makedirs(output_folder, exist_ok=True)
    for sample, (im_path, seg_path) in enumerate(zip(image_paths, seg_paths), 1):
        out_path = os.path.join(output_folder, f"S{sample}.h5")
        postprocess_segmentation(im_path, seg_path, out_path)


if __name__ == "__main__":
    main()
