import argparse
import os

from plate_utils import read_plate_config

OUTPUT_ROOT = "/scratch/pape/covid-if-2/data"


def compute_scores(folder_name):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")  # e.g. "./plate_configs/mix_wt_alpha_control.json"
    args = parser.parse_args()
    plate_config = read_plate_config(args.config_file)
    folder_name = os.path.basename(plate_config.folder).lower()
    compute_scores(folder_name)


if __name__ == "__main__":
    main()
