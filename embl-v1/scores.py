import argparse
import os
from glob import glob

import numpy as np
import pandas as pd

from plate_utils import read_plate_config, to_site_name, to_well_name

OUTPUT_ROOT = "/scratch/pape/covid-if-2/data"


# how do we measure the antibody response score for the individual patterns?
# - simple approach: median serum expression per cell divided by median spike expression
# - if we have negative control cells we could also compute a ratio to these! (makes much more sense!)
# - but how do we normalize for the spike expression?

# TODO needs to be updated for negative control cells
def _compute_scores(well_name, well_table):
    # only take into account the 'is_stained' cells
    well_table = well_table[well_table["is_stained"]]

    patterns, pattern_counts = np.unique(well_table["prediction"].values, return_counts=True)

    # TODO get the bg intensities (from some table)
    bg_spike = 0.0
    bg_serum = 0.0

    score_table = {
        "well": [well_name] * len(patterns),
        "pattern": [],
        "n_cells": [],
        "score": []
    }

    # TODO discuss with Vibor and update how the scores are computed?
    for pattern, pattern_count in zip(patterns, pattern_counts):
        score_table["pattern"].append(pattern)
        score_table["n_cells"].append(pattern_count)

        pattern_table = well_table[well_table["prediction"] == pattern]
        pattern_table["serum_median"] -= bg_serum
        pattern_table["spike_median"] -= bg_spike

        # score_per_cell = pattern_table["serum_median"] / serum_control
        score_per_cell = pattern_table["serum_median"] / pattern_table["spike_median"]
        score_table["score"].append(np.median(score_per_cell.values))

    return pd.DataFrame.from_dict(score_table)


def compute_scores(folder_name):
    table_folders = glob(os.path.join(OUTPUT_ROOT, folder_name, "tables", "cell-segmentation_*"))
    table_folders.sort()

    well_to_table = {}
    for table_folder in table_folders:
        site_name = to_site_name(os.path.basename(table_folder), None)
        well_name = to_well_name(site_name)

        default_table = pd.read_csv(os.path.join(table_folder, "default.tsv"), sep="\t")
        stat_table = pd.read_csv(os.path.join(table_folder, "statistics_cell-segmentation.tsv"), sep="\t")
        assert (default_table["label_id"] == stat_table["label_id"]).all()

        this_table = pd.concat(
            [default_table[["is_stained", "prediction"]], stat_table[["serum_median", "spike_median"]]], axis=1
        )

        if well_name in well_to_table:
            this_table = pd.concat([well_to_table[well_name], this_table], axis=0)

        well_to_table[well_name] = this_table

    # compute the actual scores
    scores = [_compute_scores(well_name, well_table) for well_name, well_table in well_to_table.items()]

    # to pandas table
    scores = pd.concat(scores, axis=0)

    print(scores)
    # TODO save as excel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")  # e.g. "./plate_configs/mix_wt_alpha_control.json"
    args = parser.parse_args()
    plate_config = read_plate_config(args.config_file)
    folder_name = os.path.basename(plate_config.folder).lower()
    compute_scores(folder_name)


if __name__ == "__main__":
    main()
