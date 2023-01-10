import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
import zarr

from skimage.transform import resize
from tqdm import tqdm
from torchvision.models.resnet import resnet18

from plate_utils import read_plate_config, to_well_name, CLASSES, OUTPUT_ROOT


# CHECKPOINT = os.path.join("/g/kreshuk/pape/Work/my_projects/covid-if-2/classification/checkpoints",
#                           "classification_v3_resnet18_with_mask")
# CHECKPOINT = os.path.join("/g/kreshuk/pape/Work/my_projects/covid-if-2/classification/checkpoints",
#                           "classification_v3_resnet18")
CHECKPOINT = os.path.join("/g/kreshuk/pape/Work/my_projects/covid-if-2/classification/checkpoints",
                          "classification_v1_augmentations")


def no_filter(position, pattern):
    return True


def training_plate_filter(position, pattern):
    if pattern != "Markers_mixed":
        idx = int(position[1:]) - 1
        idx_in_well = idx % 9
        if idx_in_well < 7:
            return True
    return False


FILTER_FUNCTIONS = {"no_filter": no_filter, "training_plate_filter": training_plate_filter, None: no_filter}


def load_model(checkpoint):
    device = torch.device("cuda")
    model = resnet18(num_classes=len(CLASSES))
    model_state = torch.load(os.path.join(checkpoint, "best_model.pt"))
    model.load_state_dict(model_state)
    model.eval()
    model.to(device)
    with_mask = "with_mask" in checkpoint
    if with_mask:
        print("Run classification WITH MASK")
    else:
        print("Run classification WITHOUT MASK")
    return model, with_mask


def classify_cells_image(model, marker_path, nuclei_path, seg_path, table_path, with_mask):
    eps = 1e-7
    device = torch.device("cuda")

    with zarr.open(marker_path, "r") as f:
        markers = f["s0"][:]
    with zarr.open(nuclei_path, "r") as f:
        nuclei = f["s0"][:]
    with zarr.open(seg_path, "r") as f:
        seg = f["s0"][:]

    cell_table = pd.read_csv(table_path, sep="\t")

    patch_shape = (152, 152)

    class_predictions = {}
    label_ids = []
    patches = []

    def _preprocess(data, normalize=True):
        data = resize(data, patch_shape, preserve_range=True)
        data = data.astype("float32")
        if normalize:
            p_dn, p_up = data.min(), data.max()
            data = (data.astype("float32") - p_dn) / (p_up - p_dn + eps)
        return data[None]

    for _, row in cell_table.iterrows():
        label_id = row.label_id
        # we skip the cells that are not stained
        if not getattr(row, "is_stained", True):
            class_predictions[label_id] = "not-classified"
        bb = np.s_[
            int(row.bb_min_y):int(row.bb_max_y),
            int(row.bb_min_x):int(row.bb_max_x)
        ]

        mask = seg[bb] == label_id
        this_markers = markers[bb]
        this_nuclei = nuclei[bb]
        if with_mask:
            this_markers[~mask] = 0
            this_nuclei[~mask] = 0
        patch = np.concatenate([
            _preprocess(this_markers),
            _preprocess(this_nuclei),
            _preprocess(mask, normalize=False)
        ], axis=0)
        assert patch.shape == (3,) + patch_shape

        label_ids.append(label_id)
        patches.append(patch)

    patches = np.array(patches)
    assert patches.ndim == 4
    # can we do this without OOM ?
    # batch_size = 256
    with torch.no_grad():
        input_ = torch.from_numpy(patches).to(device)
        predictions = model(input_)
        predictions = predictions.max(1)[1].cpu().numpy()

    assert len(predictions) == len(label_ids)
    class_predictions.update({label_id: CLASSES[class_id] for label_id, class_id in zip(label_ids, predictions)})
    class_predictions = dict(sorted(class_predictions.items()))
    assert len(class_predictions) == len(cell_table)
    cell_table = cell_table.sort_values("label_id")
    cell_table["prediction"] = list(class_predictions.values())
    cell_table.to_csv(table_path, sep="\t", index=False)


def set_to_non_classified(table_path):
    tab = pd.read_csv(table_path, sep="\t")
    tab["prediction"] = "not-classified"
    tab.to_csv(table_path, sep="\t", index=False)


def classify_cells(folder_name, filter_function):
    ds_folder = os.path.join(OUTPUT_ROOT, folder_name)
    model, with_mask = load_model(CHECKPOINT)
    print("Run classification with model from", CHECKPOINT)
    if with_mask:
        print("with mask")
    else:
        print("WITHOUT mask")

    marker_paths = glob(os.path.join(ds_folder, "images", "ome-zarr", "*marker*"))
    marker_paths.sort()
    nuclei_paths = glob(os.path.join(ds_folder, "images", "ome-zarr", "*nuclei*"))
    nuclei_paths.sort()
    cell_seg_paths = glob(os.path.join(ds_folder, "images", "ome-zarr", "*cell-segmentation*"))
    cell_seg_paths.sort()
    cell_table_paths = glob(os.path.join(ds_folder, "tables", "*cell-segmentation*"))
    cell_table_paths.sort()

    site_table = pd.read_csv(
        os.path.join(ds_folder, "tables", "sites", "default.tsv"), sep="\t"
    )

    for mp, nup, cp, ct in tqdm(zip(marker_paths, nuclei_paths, cell_seg_paths, cell_table_paths),
                                total=len(marker_paths)):

        table_path = os.path.join(ct, "default.tsv")

        # we skip all positions that were used either as training or as validation data
        position = os.path.basename(mp)[:-len(".ome.zarr")].split("_")[1]
        pattern = site_table[site_table["position"] == position]["pattern"]
        assert len(pattern) == 1
        pattern = pattern.values[0]

        if not filter_function(position, pattern):
            # set all cells to non-classified
            set_to_non_classified(table_path)
            continue

        classify_cells_image(model, mp, nup, cp, table_path, with_mask)


def analyze_classification(folder_name, plate_config):
    table_root = os.path.join(OUTPUT_ROOT, folder_name, "tables")
    table_folders = glob(os.path.join(table_root, "*cell-segmentation*"))
    table_folders.sort()

    well_stats = {}

    for table_folder in table_folders:
        source_name = os.path.basename(table_folder)
        site_name = plate_config.to_site_name(source_name, "cell-segmentation")
        well_name = to_well_name(site_name)

        table_path = os.path.join(table_folder, "default.tsv")
        table = pd.read_csv(table_path, sep="\t")

        predicted_classes, counts = np.unique(table["prediction"].values, return_counts=True)
        valid_predictions = np.isin(predicted_classes, CLASSES)
        predicted_classes, counts = predicted_classes[valid_predictions], counts[valid_predictions]
        site_stats = {cls: cnt for cls, cnt in zip(predicted_classes, counts)}

        if well_name in well_stats:
            this_stats = well_stats[well_name]
            keys = list(set(predicted_classes).union(set(this_stats.keys())))
            this_stats = {cls: this_stats.get(cls, 0) + site_stats.get(cls, 0) for cls in keys}
        else:
            this_stats = site_stats
        well_stats[well_name] = this_stats

    classification_stats = {"well": list(well_stats.keys())}
    classification_stats.update({cls: [v.get(cls, 0) for v in well_stats.values()] for cls in CLASSES})
    classification_stats = pd.DataFrame.from_dict(classification_stats)
    classification_stats = classification_stats.set_index("well")

    print("Classification statistics:")
    print(classification_stats)
    print()
    print("Normalized classification statistics:")
    norm = classification_stats.sum(axis=1)
    for i in range(len(classification_stats)):
        classification_stats.iloc[i] /= norm[i]
    classification_stats = classification_stats.round(decimals=2)
    print(classification_stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")  # e.g. "./plate_configs/mix_wt_alpha_control.json"
    parser.add_argument("-c", "--classify", type=int, default=1)
    args = parser.parse_args()

    plate_config = read_plate_config(args.config_file)
    folder_name = os.path.basename(plate_config.folder).lower()

    if bool(args.classify):
        filter_function = FILTER_FUNCTIONS[plate_config.prediction_filter_name]
        classify_cells(folder_name, filter_function=filter_function)
    analyze_classification(folder_name, plate_config)


if __name__ == "__main__":
    main()
