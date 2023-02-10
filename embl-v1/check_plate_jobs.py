import os
from glob import glob
from pathlib import Path

from plate_utils import OUTPUT_ROOT

PROCESSED_FILE = "./processed_final_plates.json"


# TODO check if the datasets were correctly processed
def check_finished_plates(config_root):
    configs = glob(os.path.join(config_root, "*.json"))
    plate_names = [Path(conf).stem for conf in configs]

    finished_plates = []
    for name in plate_names:
        plate_ds = os.path.join(OUTPUT_ROOT, name)
        if not os.path.exists(plate_ds):
            continue

    print(finished_plates)


def main():
    # check_finished_plates("./plate_configs/FINAL_DATASETS")
    check_finished_plates("./plate_configs/FINAL_DATASETS_mAB")


if __name__ == "__main__":
    check_finished_plates()
