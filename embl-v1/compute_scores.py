import argparse
import os
from glob import glob

import numpy as np
import pandas as pd

from plate_utils import read_plate_config, to_well_name, OUTPUT_ROOT

PATTERN_TO_NAME = {
    "mScarlet-Giantin": "Wildtype - Giantin",
    "LCK-mScarlet": "Delta - LCK",
    "mScarlet-H2A": "Omicron BA.1 - H2A",
    "3xNLS-mScarlet": "Nucleocapsid - 3xNLS",
    "mScarlet-Lamin": "Control - Lamin"
}


def _make_plots(score_table, well_table, res_folder):
    pass


def _insert_empty_row(table):
    new_index = pd.RangeIndex(len(table) + 1)
    new_table = pd.DataFrame(np.nan, index=new_index, columns=table.columns)
    ids = np.arange(len(table))
    new_table.loc[ids] = table.values
    return new_table


def _scores_and_plots(well_name, well_table, plate_config, res_folder, well_bg):
    spike_patterns = plate_config.spike_patterns
    assert spike_patterns is not None
    nc_patterns = plate_config.nucleocapsid_patterns
    assert nc_patterns is not None
    control_patterns = plate_config.untagged_patterns
    assert control_patterns is not None
    patterns = spike_patterns + nc_patterns + control_patterns

    # only keep the cell that have passed QC
    qc_passed = well_table["qc_passed"]
    well_table = well_table[qc_passed]

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
        "spike_median_intensity": [],
        "normalization_ratio": [],
    }

    # background subtraction
    bg_serum, bg_spike = well_bg["serum"], well_bg["spike"]
    well_table["serum_median"] -= bg_serum
    well_table["spike_median"] -= bg_spike

    def _compute_intensity(column, mask):
        intensities = well_table[column].values[mask]
        return np.mean(intensities), np.std(intensities)

    def _compute_intensity_ratio(column_nom, column_denom, correction_nom, mask):
        nominator = well_table[column_nom][mask]
        denominator = well_table[column_denom][mask]
        values = (nominator - correction_nom) / denominator
        return np.mean(values)

    control_mask = well_table["prediction"].isin(control_patterns)
    control_mean_intensity, control_mean_intensity_std = _compute_intensity("serum_mean", control_mask)
    control_median_intensity, control_median_intensity_std = _compute_intensity("serum_median", control_mask)

    def _stats_for_patterns(name, patterns, compute_norm_ratio, compute_other_ratio):
        if name == "spike":
            score_table["pattern"].append("Spike")
        else:
            score_table["pattern"].append(PATTERN_TO_NAME[name])

        pattern_mask = well_table["prediction"].isin(patterns)
        score_table["number_cells"].append(pattern_mask.sum())

        # mean based score measure
        mean_intensity, mean_intensity_std = _compute_intensity("serum_mean", pattern_mask)
        score_table["mean_intensity"].append(mean_intensity)
        score_table["mean_intensity_std"].append(mean_intensity_std)
        score_table["score_mean"].append(mean_intensity / control_mean_intensity)

        # median based score measure
        median_intensity, median_intensity_std = _compute_intensity("serum_median", pattern_mask)
        score_table["median_intensity"].append(median_intensity)
        score_table["median_intensity_std"].append(median_intensity_std)
        score_table["score_median"].append(median_intensity / control_median_intensity)

        # spike intensity
        spike_intensity, _ = _compute_intensity("spike_median", pattern_mask)
        score_table["spike_median_intensity"].append(spike_intensity)

        # ratio measures
        score_table["normalization_ratio"].append(
            _compute_intensity_ratio("serum_median", "spike_median", control_median_intensity, pattern_mask)
        )

        return score_table

    # compute statistics for the individual patterns
    for pattern in PATTERN_TO_NAME:
        score_table = _stats_for_patterns(pattern, [pattern], compute_norm_ratio=True,
                                          compute_other_ratio=pattern in nc_patterns)

    # compute combined statistcs for the spike patterns and nc patterns
    score_table = _stats_for_patterns("spike", spike_patterns, compute_norm_ratio=False, compute_other_ratio=True)

    score_table = pd.DataFrame.from_dict(score_table)
    _make_plots(score_table, well_table, res_folder)

    score_table = _insert_empty_row(score_table)
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
    ds_folder = os.path.join(OUTPUT_ROOT, folder_name)
    table_folders = glob(os.path.join(ds_folder, "tables", "cell-segmentation_*"))
    table_folders.sort()

    well_to_table = {}
    qc_failed = []
    for table_folder in table_folders:
        site_name = plate_config.to_site_name(os.path.basename(table_folder), None)
        well_name = to_well_name(site_name)

        default_table = pd.read_csv(os.path.join(table_folder, "default.tsv"), sep="\t")
        if not quality_control_image(site_name, default_table):
            qc_failed.append(site_name)
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

    well_to_bg = {}
    bg_stat_table = pd.read_csv(os.path.join(ds_folder, "tables", "sites", "bg_stats.tsv"), sep="\t")
    for _, row in bg_stat_table.iterrows():
        well = row.region_id.split("_")[0]
        if well in well_to_bg:
            bg_serum, bg_spike = well_to_bg[well]["serum"], well_to_bg[well]["spike"]
        else:
            bg_serum, bg_spike = [], []

        bg_serum.append(row.serum_median)
        bg_spike.append(row.spike_median)

        well_to_bg[well] = {"serum": bg_serum, "spike": bg_spike}

    well_to_bg = {
        well: {"serum": np.mean(well_to_bg[well]["serum"]),
               "spike": np.mean(well_to_bg[well]["spike"])}
        for well in well_to_bg
    }

    res_folder = os.path.join("analysis_results", folder_name)
    os.makedirs(res_folder, exist_ok=True)

    # compute the actual scores
    scores = [_scores_and_plots(well_name, well_table,
                                plate_config, res_folder,
                                well_to_bg[well_name])
              for well_name, well_table in well_to_table.items()]

    # to pandas table
    scores = pd.concat(scores, axis=0)

    # save as excel
    save_path = os.path.join(res_folder, f"{folder_name}.xlsx")
    print("Analysis results were saved to", save_path)
    breakpoint()
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
