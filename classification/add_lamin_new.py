import os

import pandas as pd
import numpy as np
from elf.io import open_file
# from tqdm import tqdm


MOBIE_ROOT = "/g/kreshuk/data/covid-if-2/from_nuno/mobie-tmp/data"
OUTPUT_ROOT = "/scratch/pape/covid-if-2"


def prepare_training_data(ds_name, apply_mask=False):
    ds_folder = os.path.join(MOBIE_ROOT, ds_name)
    table_folder = os.path.join(ds_folder, "tables", "sites")

    site_table = pd.read_csv(os.path.join(table_folder, "default.tsv"), sep="\t")
    # Get the image level stats so that they can be used for normalization
    # I don't think we need bg stats for normalization
    # bg_stat_table = pd.read_csv(os.path.join(table_folder, "bg_stats.tsv"), sep="\t")

    out_root = os.path.join(OUTPUT_ROOT, "training_data", "classification", "v5")

    f_train = open_file(os.path.join(out_root, "train.zarr"), "a")
    f_val = open_file(os.path.join(out_root, "val.zarr"), "a")
    f_test = open_file(os.path.join(out_root, "test.zarr"), "a")

    new_pattern = "mScarlet-Lamin"
    patterns = f_train.attrs["classes"]
    if new_pattern not in patterns:
        patterns.append(new_pattern)

    f_val.attrs["classes"] = patterns
    f_test.attrs["classes"] = patterns
    f_train.attrs["classes"] = patterns

    # we use the following train, val, test split:
    # train: positions 0-9
    # val  : positions 10-11
    # test : positions 12-13
    # (position in well)
    train_id, val_id, test_id = len(f_train), len(f_val), len(f_test)

    # for idx, row in tqdm(site_table.iterrows(), total=len(site_table), desc="Prepare training data"):
    for idx, row in site_table.iterrows():
        position = "-".join([pp.capitalize() for pp in row.position.split("-")])
        if idx < 10:
            f, sample_id = f_train, train_id
            split = "train"
        elif idx in (10, 11):
            f, sample_id = f_val, val_id
            split = "val"
        else:
            f, sample_id = f_test, test_id
            split = "test"

        class_id = patterns.index(new_pattern)

        table_path = os.path.join(ds_folder, "tables", f"cell-segmentation_{position}", "default.tsv")
        seg_table = pd.read_csv(table_path, sep="\t")

        marker_path = os.path.join(ds_folder, "images", "ome-zarr", f"marker_{position}.ome.zarr")
        marker = open_file(marker_path, "r")["s0"][:]

        nuclei_path = os.path.join(ds_folder, "images", "ome-zarr", f"nuclei_{position}.ome.zarr")
        nuclei = open_file(nuclei_path, "r")["s0"][:]

        seg_path = os.path.join(ds_folder, "images", "ome-zarr", f"cell-segmentation_{position}.ome.zarr")
        seg = open_file(seg_path, "r")["s0"][:]

        print("Position:", position, "class:", class_id)
        print("Part of the", split, "split")
        for _, seg_row in seg_table.iterrows():

            if not seg_row.is_train:
                continue

            bb = (
                slice(int(seg_row.bb_min_y), int(seg_row.bb_max_y)),
                slice(int(seg_row.bb_min_x), int(seg_row.bb_max_x)),
            )

            this_mask = seg[bb] == seg_row.label_id
            assert this_mask.sum() > 0
            inv_mask = np.logical_not(this_mask)

            this_marker = marker[bb]
            if apply_mask:
                this_marker[inv_mask] = 0

            this_nuclei = nuclei[bb]
            if apply_mask:
                this_nuclei[inv_mask] = 0

            data = np.concatenate(
                [this_marker[None],
                 this_nuclei[None],
                 this_mask[None].astype(this_marker.dtype)], axis=0
            )

            ds = f.create_dataset(f"sample{sample_id:06}", data=data, chunks=data.shape)
            ds.attrs["class_id"] = class_id
            ds.attrs["class_name"] = new_pattern
            sample_id += 1

        if idx < 10:
            print(sample_id - train_id, "new train samples")
            train_id = sample_id
        elif idx in (10, 11):
            print(sample_id - val_id, "new val samples")
            val_id = sample_id
        else:
            print(sample_id - test_id, "new test samples")
            test_id = sample_id

    print("Number of training samples:", train_id + 1)
    print("Number of validation samples:", val_id + 1)
    print("Number of test samples:", test_id + 1)


def main():
    prepare_training_data("230111_new_lamin")


if __name__ == "__main__":
    main()
