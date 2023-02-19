import os
import time

from glob import glob
from pathlib import Path
from subprocess import run


def get_plate_configs(config_root):
    configs = glob(os.path.join(config_root, "*.json"))
    n_configs = len(configs)
    plate_names = [Path(conf).stem for conf in configs]
    plate_names_lower = [name.lower() for name in plate_names]
    processed_plates = os.listdir("./analysis_results")
    configs = list(set(plate_names_lower) - set(processed_plates))
    configs = [plate_names[plate_names_lower.index(name)] for name in configs]
    print(len(configs), "/", n_configs, "plates still need to be processed")
    return configs


def submit_plates(config_root, use_gpu):
    configs = get_plate_configs(config_root)
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
    submit_plates("./plate_configs/FINAL_DATASETS", use_gpu)
    # submit_plates("./plate_configs/FINAL_DATASETS_mAB", use_gpu)


if __name__ == "__main__":
    main()
