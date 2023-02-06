import os
from glob import glob
from shutil import copytree

import pandas as pd
import numpy as np
from elf.io import open_file

CLASSES = ["3xNLS-mScarlet", "LCK-mScarlet", "mScarlet-H2A", "mScarlet-Giantin", "mScarlet-Lamin"]


def require_previous_version(output_folder, prev_version):
    prev_folder = f"/scratch/pape/covid-if-2/training_data/manual/v{prev_version}"
    for split in ("train.zarr", "test.zarr", "val.zarr"):
        out_path = os.path.join(output_folder, split)
        if os.path.exists(out_path):
            continue
        copytree(
            os.path.join(prev_folder, split),
            out_path
        )


def annotation_to_dataset_v1(input_folder, ann):
    x = os.path.splitext(os.path.basename(ann))[0].rstrip('_Labeling').replace('_', '-')
    ds_name = f"230107_mab_omscreen_Screen-Scene-{x}"
    dataset = os.path.join(input_folder, ds_name)
    if not os.path.exists(dataset):
        ds_name = f"230111_test_bindingaffinity_0.95-Scene-{x}"
        dataset = os.path.join(input_folder, ds_name)
    assert os.path.exists(dataset)
    return dataset


def annotation_to_dataset_v2(input_folder, ann):
    x = os.path.splitext(os.path.basename(ann))[0].rstrip('.txt').replace('_', '-')
    ds_name = f"230107_mab_omscreen_Screen-Scene-{x}"
    dataset = os.path.join(input_folder, ds_name)
    if not os.path.exists(dataset):
        ds_name = f"230111_test_bindingaffinity_0.95-Scene-{x}"
        dataset = os.path.join(input_folder, ds_name)
    assert os.path.exists(dataset)
    return dataset


def export_manual_annotations(input_folder, table_folder, output_folder, version):
    os.makedirs(output_folder, exist_ok=True)
    if version > 1:
        require_previous_version(output_folder, version - 1)

    if version == 1:
        annotation_to_dataset = annotation_to_dataset_v1
        n_train, n_val = 6, 1
    elif version == 2:
        annotation_to_dataset = annotation_to_dataset_v2
        n_train, n_val = 7, 2
    else:
        raise RuntimeError

    annotations = glob(os.path.join(table_folder, "*.tsv"))
    datasets = [annotation_to_dataset(input_folder, ann) for ann in annotations]

    f_train = open_file(os.path.join(output_folder, "train.zarr"), "a")
    f_train.attrs["channels"] = ["marker", "nuclei", "mask"]
    f_train.attrs["classes"] = CLASSES

    f_val = open_file(os.path.join(output_folder, "val.zarr"), "a")
    f_val.attrs["channels"] = ["marker", "nuclei", "mask"]
    f_val.attrs["classes"] = CLASSES

    f_test = open_file(os.path.join(output_folder, "test.zarr"), "a")
    f_test.attrs["channels"] = ["marker", "nuclei", "mask"]
    f_test.attrs["classes"] = CLASSES

    exported_patterns = []
    ignored_cells = 0
    total_n_cells = 0

    f_out = f_train
    sample_id = len(f_train)

    for ii, (ds, annotation_table) in enumerate(zip(datasets, annotations)):
        if ii == n_train:
            f_out = f_val
            sample_id = len(f_val)
        if ii == n_train + n_val:
            f_out = f_test
            sample_id = len(f_test)

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
            ds.attrs["class_id"] = CLASSES.index(pattern)
            sample_id += 1
            total_n_cells += 1
            exported_patterns.append(pattern)

    print("Exported a total of", total_n_cells, "cells")
    print("Classes:")
    class_ids, counts = np.unique(exported_patterns, return_counts=True)
    for class_name, count in zip(class_ids, counts):
        print(class_name, ":", count)
    print(ignored_cells, "were ignored")


def main():
    version = 2

    input_folder = f"/g/kreshuk/data/covid-if-2/training_data/v{version}"
    table_folder = f"/g/kreshuk/data/covid-if-2/training_data/v{version}-annotations"
    output_folder = f"/scratch/pape/covid-if-2/training_data/manual/v{version}"

    export_manual_annotations(input_folder, table_folder, output_folder, version)


if __name__ == "__main__":
    main()
