import h5py
import napari

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed


def run_watershed(fg, hmap, seeds):
    seed_ids = np.unique(seeds)

    # TODO shrink nuclei
    shrink_radius = 7
    shrink_mask = distance_transform_edt(seeds != 0)
    seeds_shrunk = seeds.copy()
    seeds_shrunk[shrink_mask < shrink_radius] = 0
    ids_after_shrinking = np.unique(seeds_shrunk)
    missing_ids = np.setdiff1d(seed_ids, ids_after_shrinking)
    if len(missing_ids) > 0:
        reinsert_mask = np.isin(seeds, missing_ids)
        seeds_shrunk[reinsert_mask] = seeds[reinsert_mask]

    # v = napari.Viewer()
    # v.add_image(shrink_mask)
    # v.add_labels(seeds)
    # v.add_labels(seeds_shrunk)
    # napari.run()
    # quit()

    bg_seed_id = int(np.max(seeds_shrunk) + 1)

    bg_seed_mask = (fg - hmap) < 0.5
    seeds_shrunk[bg_seed_mask] = bg_seed_id

    seg = watershed(hmap, seeds_shrunk)
    seg[seg == bg_seed_id] = 0

    return seg


def dev():
    with h5py.File("seg-test.h5", "r") as f:
        fg = f["foreground"][:]
        hmap = f["hmap"][:]
        image = f["image"][:]
        nuc = f["nuclei"][:]

    seg = run_watershed(fg, hmap, nuc)

    v = napari.Viewer()
    v.add_image(image)
    v.add_image(fg)
    v.add_image(hmap)
    v.add_labels(seg)
    v.add_labels(nuc)
    napari.run()


dev()
