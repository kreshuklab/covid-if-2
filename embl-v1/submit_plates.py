import json
import os
import time

from glob import glob
from pathlib import Path
from subprocess import run


PROCESSED_FILE = "./processed_final_plates.json"


def submit_plates(config_root):
    configs = glob(os.path.join(config_root, "*.json"))
    plate_names = [Path(conf).stem for conf in configs]
    with open(PROCESSED_FILE) as f:
        processed_plates = json.load(f)
    left_to_process = list(set(plate_names) - set(processed_plates))

    print("Submitting jobs for", len(left_to_process), "plates")
    for plate in left_to_process:
        print("Submit", plate)
        config_path = f"{config_root}/{plate}.json"
        assert os.path.exists(config_path), config_path
        run(["sbatch", "plate_job_template.batch", config_path])
        time.sleep(1)


def main():
    # submit_plates("./plate_configs/FINAL_DATASETS")
    submit_plates("./plate_configs/FINAL_DATASETS_mAB")


if __name__ == "__main__":
    main()
