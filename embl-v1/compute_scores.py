import argparse
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats.mstats import pearsonr

from plate_utils import read_plate_config, to_well_name, OUTPUT_ROOT

pd.options.mode.chained_assignment = None

PATTERN_TO_NAME = {
    "mScarlet-Giantin": "Wildtype - Giantin",
    "LCK-mScarlet": "Delta - LCK",
    "mScarlet-H2A": "Omicron BA.1 - H2A",
    "3xNLS-mScarlet": "Nucleocapsid - 3xNLS",
    "mScarlet-Lamin": "Control - Lamin"
}

# measured offset values for the relevant channels
MARKER_OFFSET = 110
SERUM_OFFSET = 160
SPIKE_OFFSET = 160


def _score_plot(score_table, save_path):
    plot_table = score_table[score_table["pattern"] != "Control - Lamin"]
    sns.barplot(data=plot_table, x="pattern", y="score")
    y_max = np.ceil(plot_table["score"].max())
    plt.ylim(1.0, y_max)
    plt.xticks(rotation=90)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _ratio_plot(score_table, save_path):
    col_name = "normalization_ratio"

    ratio_patterns = list(PATTERN_TO_NAME.values())[:3]
    plot_table = score_table[score_table["pattern"].isin(ratio_patterns)]
    plot_table[col_name] = plot_table[col_name].apply(
        lambda x: x / plot_table[col_name][0]
    )
    ax = sns.barplot(data=plot_table, x="pattern", y=col_name)
    ax.bar_label(ax.containers[0])
    plt.xticks(rotation=90)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _compute_pearson(table, pattern):
    tab = table[table["prediction"] == pattern]
    x = tab["spike_median"].values
    y = tab["serum_median"].values
    r, _ = pearsonr(x, y)
    return r


# missing from the sketch: linear fit (I wouldn't add it to avoid clutter)
def _correlation_plot(well_table, save_path):
    plot_table = well_table.copy()
    plot_table["prediction"] = plot_table["prediction"].apply(lambda x: PATTERN_TO_NAME[x])

    corr_patterns = list(PATTERN_TO_NAME.values())[:4]
    plot_table = plot_table[plot_table["prediction"].isin(corr_patterns)]

    plot_table.sort_values(by="prediction", key=lambda z: z.apply(lambda x: corr_patterns.index(x)), inplace=True)

    coefficients = {pattern: _compute_pearson(plot_table, pattern)
                    for pattern in corr_patterns}
    plot_table["prediction"] = plot_table["prediction"].apply(lambda x: f"{x}: {coefficients[x]:0.2f}")
    plot_table = plot_table.rename(columns={"prediction": "Pearson's R"})

    sns.scatterplot(
        data=plot_table, x="spike_median", y="serum_median",
        hue="Pearson's R", style="Pearson's R"
    )
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _make_plots(score_table, well_table, res_folder):
    well_name = score_table["well"][0]
    _score_plot(score_table, os.path.join(res_folder, f"{well_name}_scores.png"))
    _ratio_plot(score_table, os.path.join(res_folder, f"{well_name}_ratios.png"))
    _correlation_plot(well_table, os.path.join(res_folder, f"{well_name}_correlation.png"))


def _insert_empty_row(table):
    new_index = pd.RangeIndex(len(table) + 1)
    new_table = pd.DataFrame(np.nan, index=new_index, columns=table.columns)
    ids = np.arange(len(table))
    new_table.loc[ids] = table.values
    return new_table


def _scores_and_plots(well_name, well_table, plate_config, res_folder):
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
        "serum_intensity": [],
        "serum_intensity_std": [],
        "score": [],
        "spike_intensity": [],
        "normalization_ratio": [],
        "ratio_score": [],
    }

    # bleedthrough formula:
    # Serum_correct = (Serum_raw - 160) - ((Marker_raw - 110) x 0,015) - ((Spike_raw - 160) x 0,06)
    # background subtraction

    # Instead of using the precomputed offsets we use the directly measured ones
    # serum_offset, spike_offset, marker_offset = well_bg["serum"], well_bg["spike"], well_bg["marker"]

    serum_correction = SERUM_OFFSET +\
        (well_table.loc[:, "marker_median"] - MARKER_OFFSET) * 0.015 +\
        (well_table.loc[:, "spike_median"] - SPIKE_OFFSET) * 0.06

    well_table.loc[:, "serum_median"] -= serum_correction

    well_table.loc[:, "spike_median"] -= SPIKE_OFFSET

    def _compute_intensity(column, mask):
        intensities = well_table[column].values[mask]
        return np.median(intensities), np.std(intensities)

    def _compute_intensity_ratio(column_nom, column_denom, mask):
        nominator = well_table[column_nom][mask]
        denominator = well_table[column_denom][mask]
        values = nominator / denominator
        return np.median(values)

    control_mask = well_table["prediction"].isin(control_patterns)
    control_intensity, control_intensity_std = _compute_intensity("serum_median", control_mask)

    wt_mask = well_table["prediction"].isin(["mScarlet-Giantin"])
    normalization_ratio_wt = _compute_intensity_ratio("serum_median", "spike_median", wt_mask)

    def _stats_for_patterns(name, patterns, compute_norm_ratio, compute_other_ratio):
        if name == "spike":
            score_table["pattern"].append("Spike")
        else:
            score_table["pattern"].append(PATTERN_TO_NAME[name])

        pattern_mask = well_table["prediction"].isin(patterns)
        score_table["number_cells"].append(pattern_mask.sum())

        # median based score measure
        median_intensity, median_intensity_std = _compute_intensity("serum_median", pattern_mask)
        score_table["serum_intensity"].append(median_intensity)
        score_table["serum_intensity_std"].append(median_intensity_std)
        score_table["score"].append(median_intensity / control_intensity)

        # spike intensity
        spike_intensity, _ = _compute_intensity("spike_median", pattern_mask)
        score_table["spike_intensity"].append(spike_intensity)

        # ratio measures
        normalization_ratio = _compute_intensity_ratio("serum_median", "spike_median", pattern_mask)
        score_table["normalization_ratio"].append(normalization_ratio)
        score_table["ratio_score"].append(normalization_ratio / normalization_ratio_wt)

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
    max_num_cells = 650
    if n_cells > max_num_cells:
        print("Site:", site_name, "did not pass quality control because it contains too many cells:",
              n_cells, ">", max_num_cells)
        return False
    return True


def check_offsets(ds_folder, qc_failed):
    well_to_bg = {}
    bg_stat_table = pd.read_csv(os.path.join(ds_folder, "tables", "sites", "bg_stats.tsv"), sep="\t")
    for _, row in bg_stat_table.iterrows():
        if row.region_id in qc_failed:
            continue
        well = row.region_id.split("_")[0]
        if well in well_to_bg:
            bg_serum = well_to_bg[well]["serum"]
            bg_spike = well_to_bg[well]["spike"]
            bg_marker = well_to_bg[well]["marker"]
        else:
            bg_serum, bg_spike, bg_marker = [], [], []

        bg_serum.append(row.serum_median)
        bg_spike.append(row.spike_median)
        bg_marker.append(row.marker_median)

        well_to_bg[well] = {"serum": bg_serum, "spike": bg_spike, "marker": bg_marker}

    well_to_bg = {
        well: {"serum": np.median(well_to_bg[well]["serum"]),
               "spike": np.median(well_to_bg[well]["spike"]),
               "marker": np.median(well_to_bg[well]["marker"])}
        for well in well_to_bg
    }

    # F02 is the control well without serum staining
    if "F02" in well_to_bg:
        off_measured_serum = well_to_bg["F02"]["serum"]
        off_measured_spike = well_to_bg["F02"]["spike"]
        off_measured_marker = well_to_bg["F02"]["marker"]

        if np.abs(off_measured_serum - SERUM_OFFSET) > 10:
            print(f"Measured {off_measured_serum} and pre-defined {SERUM_OFFSET} serum offsets differ for serum.")
        if np.abs(off_measured_spike - SPIKE_OFFSET) > 10:
            print(f"Measured {off_measured_spike} and pre-defined {SPIKE_OFFSET} spike offsets differ for spike.")
        if np.abs(off_measured_marker - MARKER_OFFSET) > 10:
            print(f"Measured {off_measured_marker} and pre-defined {MARKER_OFFSET} marker offsets differ for spike.")


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
                cell_table[["serum_median", "spike_median", "marker_median"]],
            ], axis=1
        )

        if well_name in well_to_table:
            this_table = pd.concat([well_to_table[well_name], this_table], axis=0)

        well_to_table[well_name] = this_table

    check_offsets(ds_folder, qc_failed)

    res_folder = os.path.join("analysis_results", folder_name)
    os.makedirs(res_folder, exist_ok=True)

    # compute the actual scores
    scores = [_scores_and_plots(well_name, well_table, plate_config, res_folder)
              for well_name, well_table in well_to_table.items()]

    # to pandas table
    scores = pd.concat(scores, axis=0)

    # save as excel
    save_path = os.path.join(res_folder, f"{folder_name}.xlsx")
    print("Analysis results were saved to", res_folder)
    scores.to_excel(save_path, index=False)

    # TODO zip the folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")  # e.g. "./plate_configs/mix_wt_alpha_control.json"
    args = parser.parse_args()
    plate_config = read_plate_config(args.config_file)
    compute_scores(plate_config)


if __name__ == "__main__":
    main()
