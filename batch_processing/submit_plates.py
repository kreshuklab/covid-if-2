import json
import os
import time

from glob import glob
from pathlib import Path
from subprocess import run

PROCESSED_FILE = "./processed_final_plates.json"
PROCESSED_FILE_MAB = "./processed_final_plates_mAB.json"


def get_plate_configs(config_root, processed_file):
    configs = glob(os.path.join(config_root, "*.json"))

    with open(processed_file) as f:
        processed_plates = json.load(f)

    n_configs = len(configs)
    plate_names = [Path(conf).stem for conf in configs]
    configs = list(set(plate_names) - set(processed_plates))
    configs = [plate_names[plate_names.index(name)] for name in configs]

    print(len(configs), "/", n_configs, "plates still need to be processed")
    return configs


def submit_plates(config_root, use_gpu, processed_file):
    configs = get_plate_configs(config_root, processed_file)
    print("Submitting jobs for", len(configs), "plates")
    template_file = "plate_job_template_gpu.batch" if use_gpu else "plate_job_template.batch"

    for plate in configs:
        print("Submit", plate)
        config_path = f"{config_root}/{plate}.json"
        assert os.path.exists(config_path), config_path
        run(["sbatch", template_file, config_path])
        time.sleep(2)


def main():
    use_gpu = True
    submit_plates("./plate_configs/FINAL_DATASETS", use_gpu, PROCESSED_FILE)
    # submit_plates("./plate_configs/FINAL_DATASETS_mAB", use_gpu, PROCESSED_FILE_MAB)


if __name__ == "__main__":
    main()
