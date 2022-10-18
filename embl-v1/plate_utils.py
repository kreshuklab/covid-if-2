import string

# TODO read channel order from file
CHANNEL_ORDER = {0: "marker", 1: "nuclei", 2: "serum"}


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
