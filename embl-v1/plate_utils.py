import json
import string

CLASSES = ["3xNLS-mScarlet", "LCK-mScarlet", "mScarlet-H2A", "mScarlet-Lamin", "mScarlet-Giantin"]


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
        self.prediction_filter_name = plate_config["prediction_filter_name"]


def read_plate_config(path):
    return PlateConfig(path)


def to_site_name(source_name, prefix):
    pos = source_name.split("_")[1]
    pos_id = int(pos[1:]) - 1
    well_id = pos_id // 9
    pos_in_well = pos_id % 9
    # this is hardcoded to the current format for the "Markers_New" data
    return f"A{well_id:02}_{pos_in_well}"


def to_well_name(site_name):
    return site_name.split("_")[0]


def to_position(well_name):
    r, c = well_name[0], well_name[1:]
    r = string.ascii_uppercase.index(r)
    return [int(c), r]
