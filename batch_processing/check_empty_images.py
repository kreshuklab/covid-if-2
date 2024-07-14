import argparse
import json
import os
from glob import glob

from convert_images import read_czi
from tqdm import tqdm

channel_order = {0: "nuclei", 1: "serum", 2: "marker", 3: "spike"}


def check_empty_well(root, pattern):
    files = glob(os.path.join(root, pattern))
    files.sort()

    corrupted_files = []
    for ff in tqdm(files):
        try:
            read_czi(ff, channel_order, permissive=False)
        except ValueError:
            corrupted_files.append(ff)

    print("Number of corrupted files:", len(corrupted_files))
    if corrupted_files:
        with open(f"corrupted_files_{os.path.basename(root)}.json", "w") as f:
            json.dump([os.path.basename(fname) for fname in corrupted_files], f)


def main():
    parser = argparse.ArgumentParser()

    default_root = "/g/kreshuk/data/covid-if-2/from_nuno/FINAL_DATASETS/230131_NS_Plate_10C1"
    parser.add_argument("-r", "--root", default=default_root)
    parser.add_argument("-p", "--pattern", default="*-D07.czi")
    args = parser.parse_args()

    check_empty_well(args.root, args.pattern)


if __name__ == "__main__":
    main()
