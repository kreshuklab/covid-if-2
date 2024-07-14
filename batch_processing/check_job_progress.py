import json
import os
from glob import glob
from pathlib import Path

import mobie
import pandas as pd
from plate_utils import read_plate_config, write_plate_config, to_well_name, to_site_name_new, INPUT_ROOT, OUTPUT_ROOT

PROCESSED_FILE = "./processed_final_plates.json"
PROCESSED_FILE_MAB = "./processed_final_plates_mAB.json"


def _get_n_classified(plate_folder):
    table_folders = glob(os.path.join(plate_folder, "tables", "cell-segmentation*"))
    n_classified = 0
    for table_folder in table_folders:
        table_path = os.path.join(table_folder, "default.tsv")
        try:
            table = pd.read_csv(table_path, sep="\t")
        except pd.errors.EmptyDataError:
            print("Table for", plate_folder, "is empty")
            n_classified += 1
            continue
        if "prediction" in table.columns:
            n_classified += 1
    return n_classified


def _get_n_qc(plate_folder):
    table_folders = glob(os.path.join(plate_folder, "tables", "cell-segmentation*"))
    n_qc = 0
    for table_folder in table_folders:
        table_path = os.path.join(table_folder, "default.tsv")
        try:
            table = pd.read_csv(table_path, sep="\t")
        except pd.errors.EmptyDataError:
            print("Table for", plate_folder, "is empty")
            n_qc += 1
            continue
        if "qc_passed" in table.columns:
            n_qc += 1
    return n_qc


def _get_expected_wells(plate_config, plate_folder):
    well_names = []

    table_folders = glob(os.path.join(plate_folder, "tables", "cell-segmentation*"))
    table_folders.sort()

    for table_folder in table_folders:
        name = os.path.basename(table_folder)
        site_name = to_site_name_new(name, prefix="cell-segmentation_")
        well_names.append(to_well_name(site_name))
    return set(well_names)


def check_progress(plate_name, plate_input_name, config_path):
    plate_config = read_plate_config(config_path)

    plate_folder = os.path.join(OUTPUT_ROOT, plate_name)
    if not os.path.exists(plate_folder):
        print(plate_name, "has not been processed at all")
        return False

    n_images = len(glob(os.path.join(INPUT_ROOT, plate_input_name, "*.czi")))
    meta = mobie.metadata.read_dataset_metadata(plate_folder)

    sources = meta["sources"]
    source_names = list(sources.keys())

    # check convert_images
    image_prefixes = ["marker", "nuclei", "serum", "spike"]
    for prefix in image_prefixes:
        n_prefix = len([name for name in source_names if prefix in name])
        if n_prefix != n_images:
            print(plate_name, ": only", n_prefix, "/", n_images, "images exist for ", prefix)
            return False
    plate_config.processed["convert_images"] = True

    # check segment_nuclei
    nuc_prefix = "nucleus-segmentation"
    n_prefix = len([name for name in source_names if nuc_prefix in name])
    if n_prefix != n_images:
        print(plate_name, ": only", n_prefix, "/", n_images, "nucleus segmentations")
        write_plate_config(config_path, plate_config)
        return False
    plate_config.processed["segment_nuclei"] = True

    # check segment_cells
    seg_prefix = "cell-segmentation"
    n_prefix = len([name for name in source_names if seg_prefix in name])
    if n_prefix != n_images:
        print(plate_name, ": only", n_prefix, "/", n_images, "cell segmentations")
        write_plate_config(config_path, plate_config)
        return False
    plate_config.processed["segment_cells"] = True

    # check compute_intensities
    site_stat_table = os.path.join(plate_folder, "tables", "sites", "bg_stats.tsv")
    if not os.path.exists(site_stat_table):
        print(plate_name, ": intensities have not been computed")
        write_plate_config(config_path, plate_config)
        return False
    plate_config.processed["compute_intensities"] = True

    # check classifiy_cells
    n_classified = _get_n_classified(plate_folder)
    if n_classified != n_images:
        print(plate_name, ": only", n_classified, "/", n_images, "are classified")
        write_plate_config(config_path, plate_config)
        return False
    plate_config.processed["classify_cells"] = True

    # check compute cell qc
    n_qc = _get_n_qc(plate_folder)
    if n_qc != n_images:
        print(plate_name, ": only", n_qc, "/", n_images, "have cell qc")
        write_plate_config(config_path, plate_config)
        return False
    plate_config.processed["compute_cell_qc"] = True

    # check compute scores 1
    output_table = f"./analysis_results/{plate_name}/{plate_name}.xlsx"
    if not os.path.exists(output_table):
        print(plate_name, ": analysis results do not exist")
        write_plate_config(config_path, plate_config)
        return False

    # check compute scores 2
    table = pd.read_excel(output_table)
    wells_in_table = set(pd.unique(table["well"]).tolist())
    expected_wells = _get_expected_wells(plate_config, plate_folder)
    missing_wells = list(expected_wells - wells_in_table)
    if missing_wells:
        print(plate_name, ": analysis results are missing for", len(missing_wells), "wells")
        write_plate_config(config_path, plate_config)
        return False

    plate_config.processed["compute_scores"] = True
    write_plate_config(config_path, plate_config)
    return True


def check_all_plates():
    if "mAB" in INPUT_ROOT:
        config_root = "./plate_configs/FINAL_DATASETS_mAB"
        processed_file = PROCESSED_FILE_MAB
    else:
        config_root = "./plate_configs/FINAL_DATASETS"
        processed_file = PROCESSED_FILE

    configs = glob(os.path.join(config_root, "*.json"))
    complete_plates = []
    for conf in configs:
        name = Path(conf).stem
        is_complete = check_progress(name.lower(), name, conf)
        if is_complete:
            complete_plates.append(name)

    with open(processed_file, "w") as f:
        json.dump(complete_plates, f)


def main():
    check_all_plates()


if __name__ == "__main__":
    main()
