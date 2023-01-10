import argparse
import os
from glob import glob

import numpy as np
import pandas as pd

from plate_utils import read_plate_config, to_well_name

# OUTPUT_ROOT = "/scratch/pape/covid-if-2/data"
OUTPUT_ROOT = "/g/kreshuk/data/covid-if-2/from_nuno/mobie-tmp/data"


def _qc_cell_absolute(table, patterns, column, threshold, op=np.greater, verbose=False):
    pattern_mask = table["prediction"].isin(patterns)
    intensity_mask = op(table[column].values, threshold)

    # filter cells that are part of the current patterns (pattern_mask: True)
    # and do not meet the intensity criterion (intensity_mask: False).
    # Resulting in the following assignment table (pm = pattern_mask, im = intensity_mask)
    # pm | im -> ASS
    #  0 |  0 -> 1
    #  0 |  1 -> 1
    #  1 |  0 -> 0
    #  1 |  1 -> 1
    mask = ~(pattern_mask & ~intensity_mask)
    table = table[mask]

    if verbose:
        print((mask).sum(), "/", len(mask), "cells meet threshold", threshold, "in", column)
        filtered_pattern_mask = table["prediction"].isin(patterns)
        print(filtered_pattern_mask.sum(), "cells of patterns", patterns, "are left")

    return table


def _qc_cell_percentile(table, patterns, column, threshold, op=np.greater, verbose=False):
    pattern_mask = table["prediction"].isin(patterns)
    absolute_threshold = np.percentile(table[column].values[pattern_mask], threshold)
    return _qc_cell_absolute(table, patterns, column, absolute_threshold, op=op, verbose=verbose)


def _compute_scores(well_name, well_table, plate_config):
    spike_patterns = plate_config.spike_patterns
    assert spike_patterns is not None
    nc_patterns = plate_config.nucleocapsid_patterns
    assert nc_patterns is not None
    untagged_patterns = plate_config.untagged_patterns
    assert untagged_patterns is not None
    patterns = spike_patterns + nc_patterns + untagged_patterns

    #
    # perform quality control for the cells:
    #
    verbose = False
    if verbose:
        print("QC for well:", well_name)

    # 1.) only keep top 75% of spike expressing cells
    well_table = _qc_cell_percentile(well_table, spike_patterns, "spike_median", threshold=25, verbose=verbose)

    # 2.) only keep top 75% of nucleocapsid expressing cells
    well_table = _qc_cell_percentile(well_table, nc_patterns, "spike_median", threshold=25, verbose=verbose)

    # 3.) filter background cells that have express any spike pattern
    well_table = _qc_cell_absolute(well_table, untagged_patterns, "spike_median", threshold=300, op=np.less,
                                   verbose=verbose)

    # 4.) filter cells based on the marker expressions for each individual pattern,
    # with empirically determined threshold values
    well_table = _qc_cell_absolute(well_table, ["LCK-mScarlet"], "marker_median", threshold=200,
                                   verbose=verbose)
    well_table = _qc_cell_absolute(well_table, ["mScarlet-H2A"], "marker_median_nucleus", threshold=450,
                                   verbose=verbose)
    well_table = _qc_cell_absolute(well_table, ["3xNLS-mScarlet"], "marker_median_nucleus", threshold=500,
                                   verbose=verbose)
    well_table = _qc_cell_absolute(well_table, ["mScarlet-Lamin"], "marker_median_nucleus", threshold=250,
                                   verbose=verbose)
    # mScarlet-Giantin is not thresholded

    if verbose:
        print()

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

        nucleus_table = pd.read_csv(os.path.join(table_folder, "statistics_nucleus-segmentation.tsv"), sep="\t")
        assert (default_table["label_id"] == cell_table["label_id"]).all()

        this_table = pd.concat(
            [
                default_table[["prediction"]],
                cell_table[["serum_median", "spike_median", "marker_median", "serum_mean"]],
                nucleus_table[["marker_median"]].rename(
                    columns={"marker_median": "marker_median_nucleus"}
                )
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
