import os
import json
from copy import deepcopy

REFERENCE_CONFIG = "./plate_configs/new_complete_screen.json"


def create_final_configs(root_dir, config_dir, reference_config):
    os.makedirs(config_dir, exist_ok=True)
    folder_names = os.listdir(root_dir)
    for fname in folder_names:
        config = deepcopy(reference_config)
        config["folder"] = fname
        config_path = os.path.join(config_dir, f"{fname}.json")
        with open(config_path, "w") as f:
            json.dump(config, f)


def main():
    with open(REFERENCE_CONFIG) as f:
        reference_config = json.load(f)
    create_final_configs(
        "/g/kreshuk/data/covid-if-2/from_nuno/FINAL_DATASETS", "./plate_configs/FINAL_DATASETS", reference_config
    )
    create_final_configs(
        "/g/kreshuk/data/covid-if-2/from_nuno/FINAL_DATASETS_mAB", "./plate_configs/FINAL_DATASETS_mAB", reference_config
    )


if __name__ == "__main__":
    main()
