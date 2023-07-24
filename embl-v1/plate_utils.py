import inspect
import json
import string

# classes and order for the v1 models
# CLASSES = ["3xNLS-mScarlet", "LCK-mScarlet", "mScarlet-H2A", "mScarlet-Lamin", "mScarlet-Giantin"]
# classes and order for the v2 models
# CLASSES = ["3xNLS-mScarlet", "LCK-mScarlet", "mScarlet-H2A", "mScarlet-Lamin", "mScarlet-Giantin", "untagged"]
# classes and order for the v3 and v5 models
CLASSES = ["3xNLS-mScarlet", "LCK-mScarlet", "mScarlet-H2A", "mScarlet-Giantin", "mScarlet-Lamin"]
# classes and order for the v4 model
# CLASSES = ["Giantin", "LCK", "H2A", "3xNLS", "Lamin"]

# INPUT_ROOT = "/g/kreshuk/data/covid-if-2/from_nuno"
INPUT_ROOT = "/g/kreshuk/data/covid-if-2/from_nuno/FINAL_DATASETS"
# INPUT_ROOT = "/g/kreshuk/data/covid-if-2/from_nuno/FINAL_DATASETS_mAB"
# INPUT_ROOT = "/g/kreshuk/data/covid-if-2/from_nuno/from_Vibor"

OUTPUT_ROOT = "/scratch/pape/covid-if-2/data"

TASKS = ["convert_images", "segment_nuclei", "segment_cells",
         "compute_intensities", "classify_cells", "compute_cell_qc",
         "compute_scores"]


def to_site_name(source_name, prefix):
    pos = source_name.split("_")[1]
    pos_id = int(pos[1:]) - 1
    well_id = pos_id % 9
    pos_in_well = pos_id % 9
    # this is hardcoded to the current format for the "Markers_New" data
    return f"A{well_id:02}_{pos_in_well}"


def to_site_name_new(source_name, prefix):
    pos = source_name.split("_")[1].split("-")
    well_name = pos[-1]
    pos_in_well = int(pos[-2][1:])
    # assert 0 <= pos_in_well <= 7
    return f"{well_name}_{pos_in_well}"


SITE_NAMES = {
    "to_site_name": to_site_name,
    "to_site_name_new": to_site_name_new,
}


# could be a data class
class PlateConfig:
    def __init__(self, path):
        with open(path, "r") as f:
            plate_config = json.load(f)
        self.folder = plate_config["folder"]
        self.nested = plate_config["nested"]

        channel_order = {int(k): v for k, v in plate_config["channel_order"].items()}
        self.channel_order = channel_order
        self.channel_colors = plate_config["channel_colors"]

        self.prediction_filter_name = plate_config.get("prediction_filter_name", "no_filter")
        self.to_site_name = SITE_NAMES[plate_config.get("to_site_name", "to_site_name_new")]

        self.spike_patterns = plate_config.get("spike_patterns", None)
        self.nucleocapsid_patterns = plate_config.get("nucleocapsid_patterns", None)
        self.untagged_patterns = plate_config.get("untagged_patterns", None)

        self.marker_correction = plate_config.get("marker_correction", 0.015)
        self.spike_correction = plate_config.get("spike_correction", 0.06)

        self.marker_offset = plate_config.get("marker_offset", 110)
        self.serum_offset = plate_config.get("serum_offset", 160)
        self.spike_offset = plate_config.get("spike_offset", 160)

        self.lamin_spike_threshold = plate_config.get("lamin_spike_threshold", 300)
        self.lck_marker_threshold = plate_config.get("lck_marker_threshold", 200)
        self.h2a_marker_threshold = plate_config.get("h2a_marker_threshold", 450)
        self.lamin_marker_threshold = plate_config.get("lamin_marker_threshold", 250)
        self.filter_saturated_nls = plate_config.get("filter_saturated_nls", False)

        # fill the process status
        self.processed = {}
        processed = plate_config.get("processed", {})
        for task in TASKS:
            self.processed[task] = processed.get(task, False)


def read_plate_config(path):
    return PlateConfig(path)


def write_plate_config(path, plate_config):
    conf = inspect.getmembers(plate_config)
    conf = {co[0]: co[1] for co in conf if not co[0].startswith("_")}
    conf = {k: v.__name__ if callable(v) else v for k, v in conf.items()}
    with open(path, "w") as f:
        json.dump(conf, f, indent=4, sort_keys=True)


def to_well_name(site_name):
    return site_name.split("_")[0]


def to_position(well_name):
    r, c = well_name[0], well_name[1:]
    r = string.ascii_uppercase.index(r)
    return [int(c), r]
