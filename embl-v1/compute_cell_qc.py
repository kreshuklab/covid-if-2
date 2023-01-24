import argparse
import os
from glob import glob

import numpy as np
import pandas as pd

from tqdm import tqdm
from plate_utils import read_plate_config, OUTPUT_ROOT


def _qc_cell_absolute(qc_input, qc_result, patterns, column, threshold, reason,
                      op=np.greater, verbose=False):
    pattern_mask = qc_input["prediction"].isin(patterns)
    intensity_mask = op(qc_input[column].values, threshold)

    if qc_result is None:
        qc_mask = np.full(len(pattern_mask), True, dtype="bool")
        qc_reason = np.array([""] * len(pattern_mask))
    else:
        qc_mask = qc_result["qc_passed"].values
        qc_reason = qc_result["qc_reason"].values

    # filter cells that are part of the current patterns (pattern_mask: True)
    # and do not meet the intensity criterion (intensity_mask: False).
    # Resulting in the following assignment table (pm = pattern_mask, im = intensity_mask)
    # pm | im -> QC-failed
    #  0 |  0 -> 0
    #  0 |  1 -> 0
    #  1 |  0 -> 1
    #  1 |  1 -> 0
    qc_failed = pattern_mask & ~intensity_mask

    # set qc failed to false
    qc_mask[qc_failed] = False

    # update the qc reason
    qc_reason = np.array([qr + f"{reason}, " if qf else qr for qr, qf in zip(qc_reason, qc_failed)])

    if verbose:
        print(
            (~qc_failed).sum(), "/", len(qc_failed), "cells have passed qc:", threshold, "in", column, "for", patterns
        )

    qc_result = pd.DataFrame.from_dict(
        {"label_id": qc_input["label_id"].values,
         "qc_passed": qc_mask,
         "qc_reason": qc_reason}
    )
    return qc_result


def _qc_cell_percentile(qc_input, qc_result, patterns, column, threshold, reason,
                        op=np.greater, verbose=False):
    pattern_mask = qc_input["prediction"].isin(patterns)
    if pattern_mask.sum() == 0:
        return qc_result
    absolute_threshold = np.percentile(qc_input[column].values[pattern_mask], threshold)
    return _qc_cell_absolute(qc_input, qc_result, patterns, column, absolute_threshold, reason,
                             op=op, verbose=verbose)


def cell_qc_site(plate_config, table_folder, verbose):
    spike_patterns = plate_config.spike_patterns
    assert spike_patterns is not None
    nc_patterns = plate_config.nucleocapsid_patterns
    assert nc_patterns is not None
    untagged_patterns = plate_config.untagged_patterns
    assert untagged_patterns is not None
    all_patterns = spike_patterns + nc_patterns + untagged_patterns

    default_table_path = os.path.join(table_folder, "default.tsv")
    default_table = pd.read_csv(default_table_path, sep="\t")

    cell_table = pd.read_csv(os.path.join(table_folder, "statistics_cell-segmentation.tsv"), sep="\t")
    assert (default_table["label_id"] == cell_table["label_id"]).all()

    nucleus_table = pd.read_csv(os.path.join(table_folder, "statistics_nucleus-segmentation.tsv"), sep="\t")
    assert (default_table["label_id"] == nucleus_table["label_id"]).all()

    qc_input = pd.concat(
        [
            default_table[["label_id", "prediction"]],
            cell_table[["serum_median", "spike_median", "marker_median",
                        "serum_mean", "size"]],
            nucleus_table[["marker_median", "size", "marker_saturation_ratio"]].rename(
                columns={"marker_median": "marker_median_nucleus",
                         "size": "size_nucleus",
                         "marker_saturation_ratio": "marker_saturation_ratio_nucleus"}
            )
        ], axis=1
    )

    #
    # perform quality control for the cells:
    #

    # 1.) only keep top 75% of the expression in the spike channel
    # for any spike or nucleocapsid expressing pattern
    qc_result = None
    for pattern in spike_patterns + nc_patterns:
        qc_result = _qc_cell_percentile(qc_input, qc_result, [pattern], "spike_median", threshold=25,
                                        verbose=verbose, reason="Intensity in spike channel too low.")

    # 2.) filter background cells that express spike pattern (have intensity < 300 in spike channel)
    qc_result = _qc_cell_absolute(qc_input, qc_result, untagged_patterns, "spike_median", threshold=300, op=np.less,
                                  verbose=verbose, reason="Spike intensity in BG Cell")

    # 3.) size based thresholding
    qc_result = _qc_cell_absolute(qc_input, qc_result, all_patterns, "size", op=np.greater, threshold=5000,
                                  reason="cell too small")
    qc_result = _qc_cell_absolute(qc_input, qc_result, all_patterns, "size", op=np.less, threshold=85000,
                                  reason="cell too large")

    # 4.) filter cells based on the marker expressions for each individual pattern,
    # with empirically determined threshold values
    qc_result = _qc_cell_absolute(qc_input, qc_result, ["LCK-mScarlet"], "marker_median", threshold=200,
                                  verbose=verbose, reason="Marker intensity LCK")
    qc_result = _qc_cell_absolute(qc_input, qc_result, ["mScarlet-H2A"], "marker_median_nucleus", threshold=450,
                                  verbose=verbose, reason="Marker intensity H2A")
    qc_result = _qc_cell_absolute(qc_input, qc_result, ["3xNLS-mScarlet"], "marker_median_nucleus", threshold=500,
                                  verbose=verbose, reason="Marker intensity 3xNLS")
    qc_result = _qc_cell_absolute(qc_input, qc_result, ["mScarlet-Lamin"], "marker_median_nucleus", threshold=250,
                                  verbose=verbose, reason="Marker intensity Lamin")
    # mScarlet-Giantin is not thresholded

    # .) filter saturated H2A cells
    qc_result = _qc_cell_absolute(qc_input, qc_result, ["mScarlet-H2A"], "marker_saturation_ratio_nucleus",
                                  threshold=0.15, op=np.less, verbose=verbose, reason="H2A Marker saturation")

    if verbose:
        print("In total", qc_result["qc_passed"].values.sum(), "/", len(qc_result), "cells have passed QC.")
        print()

    # append the qc columns to the default table and resave it
    assert (default_table["label_id"] == qc_result["label_id"]).all()
    default_table["qc_passed"] = qc_result["qc_passed"]
    default_table["qc_reason"] = qc_result["qc_reason"]
    default_table.to_csv(default_table_path, sep="\t", index=False)


def compute_cell_qc(plate_config, verbose):
    ds_name = os.path.basename(plate_config.folder).lower()
    table_folders = glob(os.path.join(OUTPUT_ROOT, ds_name, "tables", "cell-segmentation_*"))
    table_folders.sort()
    for table_folder in tqdm(table_folders):
        cell_qc_site(plate_config, table_folder, verbose)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")  # e.g. "./plate_configs/mix_wt_alpha_control.json"
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    plate_config = read_plate_config(args.config_file)
    compute_cell_qc(plate_config, args.verbose)


if __name__ == "__main__":
    main()
