import os
from glob import glob
from pathlib import Path

from tqdm import tqdm

from plate_utils import INPUT_ROOT, read_plate_config
from compute_cell_qc import compute_cell_qc
from compute_scores import compute_scores


def get_plate_names(config_root):
    config_files = glob(os.path.join(config_root, "*.json"))
    plate_names = [Path(ff).stem for ff in config_files]
    return plate_names


def recompute_all_qc(config_root, plate_names):
    for name in tqdm(plate_names, desc="Recompute cell QC"):
        config = os.path.join(config_root, f"{name}.json")
        plate_config = read_plate_config(config)
        compute_cell_qc(plate_config, verbose=False)


def recompute_all_scores(config_root, plate_names):
    for name in tqdm(plate_names, desc="Recompute analysis"):
        config = os.path.join(config_root, f"{name}.json")
        plate_config = read_plate_config(config)
        compute_scores(plate_config)


def main():
    if "mAB" in INPUT_ROOT:
        config_root = "./plate_configs/FINAL_DATASETS_mAB"
    elif "Vibor" in INPUT_ROOT:
        config_root = "./plate_configs/vibor"
    else:
        config_root = "./plate_configs/FINAL_DATASETS"

    plate_names = get_plate_names(config_root)
    recompute_qc = False
    if recompute_qc:
        recompute_all_qc(config_root, plate_names)
    recompute_all_scores(config_root, plate_names)


if __name__ == "__main__":
    main()
