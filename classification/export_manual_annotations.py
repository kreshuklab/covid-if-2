import os
from glob import glob

import pandas as pd
import numpy as np
from elf.io import open_file


def export_manual_annotations(input_folder, table_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    annotations = glob(os.path.join(table_folder, "*.tsv"))

    ds_names = [
        f"230107_mab_omscreen_Screen-Scene-{os.path.splitext(os.path.basename(ann))[0].rstrip('_Labeling').replace('_', '-')}"
        for ann in annotations
    ]
    datasets = [os.path.join(input_folder, ds_name) for ds_name in ds_names]
    assert all(os.path.exists(ds) for ds in datasets)

    output_path = os.path.join(output_folder, "data.zarr")
    f_out = open_file(output_path, "a")
    f_out.attrs["channels"] = ["marker", "nuclei", "mask"]

    exported_patterns = []
    ignored_cells = 0
    sample_id = 0
    for ds, annotation_table in zip(datasets, annotations):
        annotation_table = pd.read_csv(annotation_table, sep="\t")

        table_path = os.path.join(ds, "tables", "cell-segmentation", "default.tsv")
        assert os.path.exists(table_path)
        table = pd.read_csv(table_path, sep="\t")
        assert (annotation_table.label_id == table.label_id).all()

        marker_path = os.path.join(ds, "images", "ome-zarr", "marker.ome.zarr")
        assert os.path.exists(marker_path)
        marker = open_file(marker_path, "r")["s0"][:]

        nuclei_path = os.path.join(ds, "images", "ome-zarr", "nuclei.ome.zarr")
        assert os.path.exists(nuclei_path)
        nuclei = open_file(nuclei_path, "r")["s0"][:]

        seg_path = os.path.join(ds, "images", "ome-zarr", "cell-segmentation.ome.zarr")
        assert os.path.exists(seg_path)
        seg = open_file(seg_path, "r")["s0"][:]

        for i, seg_row in table.iterrows():

            annotation_row = annotation_table.iloc[i]
            assert annotation_row.label_id == seg_row.label_id
            pattern = annotation_row.prediction

            if pattern == "ignore":
                ignored_cells += 1
                continue

            bb = (
                slice(int(seg_row.bb_min_y), int(seg_row.bb_max_y)),
                slice(int(seg_row.bb_min_x), int(seg_row.bb_max_x)),
            )

            this_mask = seg[bb] == seg_row.label_id
            assert this_mask.sum() > 0
            this_marker = marker[bb]
            this_nuclei = nuclei[bb]

            data = np.concatenate(
                [this_marker[None],
                 this_nuclei[None],
                 this_mask[None].astype(this_marker.dtype)], axis=0
            )

            ds = f_out.create_dataset(f"sample{sample_id:06}", data=data, chunks=data.shape)
            ds.attrs["class_name"] = pattern
            sample_id += 1
            exported_patterns.append(pattern)

    print("Exported a total of", sample_id, "cells")
    print("Classes:")
    class_ids, counts = np.unique(exported_patterns, return_counts=True)
    for class_name, count in zip(class_ids, counts):
        print(class_name, ":", count)
    print(ignored_cells, "were ignored")


# TODO enable test train val splits
def main():
    version = 1

    input_folder = f"/g/kreshuk/data/covid-if-2/training_data/v{version}"
    table_folder = f"/g/kreshuk/data/covid-if-2/training_data/v{version}-annotations"
    output_folder = f"/scratch/pape/covid-if-2/training_data/manual/v{version}"

    export_manual_annotations(input_folder, table_folder, output_folder)


if __name__ == "__main__":
    main()
