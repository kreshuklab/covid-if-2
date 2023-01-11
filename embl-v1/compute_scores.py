import argparse
import os
from glob import glob

import numpy as np
import pandas as pd

from plate_utils import read_plate_config, to_well_name, OUTPUT_ROOT


def _compute_scores(well_name, well_table, plate_config):
    spike_patterns = plate_config.spike_patterns
    assert spike_patterns is not None
    nc_patterns = plate_config.nucleocapsid_patterns
    assert nc_patterns is not None
    untagged_patterns = plate_config.untagged_patterns
    assert untagged_patterns is not None
    patterns = spike_patterns + nc_patterns + untagged_patterns

    # only keep the cell that have passed QC
    qc_passed = well_table["qc_passed"]
    well_table = well_table[qc_passed]

    # for now, we don't do any background subtraction
    # bg_serum = 0.0
    # bg_spike = 0.0

    score_table = {
        "well": (len(patterns) + 1) * [well_name],
        "pattern": [],
        "number_cells": [],
        "mean_intensity": [],
        "mean_intensity_std": [],
        "score_mean": [],
        "median_intensity": [],
        "median_intensity_std": [],
        "spike_intensity": [],
        "score_median": [],
        "normalization_ratio": [],
        "other_ratio": [],
    }

    def _compute_intensity(column, mask):
        intensities = well_table[column].values[mask]
        return np.mean(intensities), np.std(intensities)

    bg_mask = well_table["prediction"].isin(untagged_patterns)
    bg_mean_intensity, bg_mean_intensity_std = _compute_intensity("serum_mean", bg_mask)
    bg_median_intensity, bg_median_intensity_std = _compute_intensity("serum_median", bg_mask)

    def _stats_for_patterns(name, patterns, compute_norm_ratio, compute_other_ratio):
        score_table["pattern"].append(name)

        pattern_mask = well_table["prediction"].isin(patterns)
        score_table["number_cells"].append(pattern_mask.sum())

        # mean based score measure
        mean_intensity, mean_intensity_std = _compute_intensity("serum_mean", pattern_mask)
        score_table["mean_intensity"].append(mean_intensity)
        score_table["mean_intensity_std"].append(mean_intensity_std)
        score_table["score_mean"].append(mean_intensity / bg_mean_intensity)

        # median based score measure
        median_intensity, median_intensity_std = _compute_intensity("serum_median", pattern_mask)
        score_table["median_intensity"].append(median_intensity)
        score_table["median_intensity_std"].append(median_intensity_std)
        score_table["score_median"].append(median_intensity / bg_median_intensity)

        # spike intensity
        spike_intensity, _ = _compute_intensity("spike_median", pattern_mask)
        score_table["spike_intensity"].append(spike_intensity)

        # ratio measures
        if compute_norm_ratio:
            spike_intensity, _ = _compute_intensity("spike_median", pattern_mask)
            score_table["normalization_ratio"].append(
                (median_intensity - bg_median_intensity) / spike_intensity
            )
        else:
            score_table["normalization_ratio"].append(None)

        if compute_other_ratio:
            score_table["other_ratio"].append(
                (median_intensity - bg_median_intensity) / bg_median_intensity
            )
        else:
            score_table["other_ratio"].append(None)

        return score_table

    # compute statistics for the individual patterns
    for pattern in patterns:
        score_table = _stats_for_patterns(pattern, [pattern], compute_norm_ratio=True,
                                          compute_other_ratio=pattern in nc_patterns)

    # compute combined statistcs for the spike patterns and nc patterns
    score_table = _stats_for_patterns("spike", spike_patterns, compute_norm_ratio=False, compute_other_ratio=True)

    score_table = pd.DataFrame.from_dict(score_table)
    return score_table


def quality_control_image(site_name, table):
    n_cells = len(table)
    min_num_cells = 35
    if n_cells < min_num_cells:
        print("Site:", site_name, "did not pass quality control because it contains too few cells:",
              n_cells, "<", min_num_cells)
        return False
    max_num_cells = 700
    if n_cells > max_num_cells:
        print("Site:", site_name, "did not pass quality control because it contains too many cells:",
              n_cells, ">", max_num_cells)
        return False
    return True


def compute_scores(plate_config):
    folder_name = os.path.basename(plate_config.folder).lower()
    table_folders = glob(os.path.join(OUTPUT_ROOT, folder_name, "tables", "cell-segmentation_*"))
    table_folders.sort()

    well_to_table = {}
    for table_folder in table_folders:
        site_name = plate_config.to_site_name(os.path.basename(table_folder), None)
        well_name = to_well_name(site_name)

        default_table = pd.read_csv(os.path.join(table_folder, "default.tsv"), sep="\t")
        if not quality_control_image(site_name, default_table):
            continue

        cell_table = pd.read_csv(os.path.join(table_folder, "statistics_cell-segmentation.tsv"), sep="\t")
        assert (default_table["label_id"] == cell_table["label_id"]).all()

        this_table = pd.concat(
            [
                default_table[["prediction", "qc_passed"]],
                cell_table[["serum_median", "serum_mean", "spike_median"]],
            ], axis=1
        )

        if well_name in well_to_table:
            this_table = pd.concat([well_to_table[well_name], this_table], axis=0)

        well_to_table[well_name] = this_table

    # compute the actual scores
    scores = [_compute_scores(well_name, well_table, plate_config) for well_name, well_table in well_to_table.items()]

    # to pandas table
    scores = pd.concat(scores, axis=0)

    # save as excel
    res_folder = "analysis_results"
    save_path = os.path.join(res_folder, f"{folder_name}.xlsx")
    print("Analysis results were saved to", save_path)
    os.makedirs(res_folder, exist_ok=True)
    scores.to_excel(save_path, index=False)

    # TODO save in MoBIE format


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")  # e.g. "./plate_configs/mix_wt_alpha_control.json"
    args = parser.parse_args()
    plate_config = read_plate_config(args.config_file)
    compute_scores(plate_config)


if __name__ == "__main__":
    main()
