import json
import os

from plate_utils import INPUT_ROOT, read_plate_config
from compute_scores import compute_scores

PROCESSED_FILE = "./processed_final_plates.json"
PROCESSED_FILE_MAB = "./processed_final_plates_mAB.json"


def recompute_all_scores(config_root, processed_file):
    with open(processed_file) as f:
        plate_names = json.load(f)

    for name in plate_names:
        config = os.path.join(config_root, f"{name}.json")
        plate_config = read_plate_config(config)
        compute_scores(plate_config)


def main():
    if "mAB" in INPUT_ROOT:
        config_root = "./plate_configs/FINAL_DATASETS_mAB"
        processed_file = PROCESSED_FILE_MAB
    else:
        config_root = "./plate_configs/FINAL_DATASETS"
        processed_file = PROCESSED_FILE

    recompute_all_scores(config_root, processed_file)


if __name__ == "__main__":
    main()
