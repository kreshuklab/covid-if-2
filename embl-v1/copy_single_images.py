import os
from glob import glob

import mobie
import numpy as np

ROOT = "/g/kreshuk/data/covid-if-2/from_nuno/mobie-tmp/data"
OUT_ROOT = "./for-annotation"


def copy_well(dataset, well, pos_per_well):
    image_in_folder = os.path.join(ROOT, dataset, "images", "ome-zarr")
    assert os.path.exists(image_in_folder), image_in_folder
    image_paths = glob(os.path.join(image_in_folder, f"*{well}*"))
    image_paths.sort()

    image_names = [os.path.basename(path) for path in image_paths]
    name_to_pos = {name: name.split("_")[-1][:-9] for name in image_names}
    pos_to_names = {}
    for name, pos in name_to_pos.items():
        if pos in pos_to_names:
            names = pos_to_names[pos]
        else:
            names = []
        names.append(name)
        pos_to_names[pos] = names

    # TODO improve pos sampling
    sampled_pos = np.random.choice(list(pos_to_names.keys()), size=2, replace=False)
    # sampled_pos = ["Screen-Scene-232-P1-F03"]

    for pos in sampled_pos:
        ds_name = f"{dataset}_{pos}"
        image_names = pos_to_names[pos][::-1]

        ds_folder = os.path.join(OUT_ROOT, ds_name)
        for name in image_names:
            path = os.path.join(image_in_folder, name)
            im_name = name.split("_")[0]
            tmp_folder = f"./tmps/{ds_name}_{im_name}"
            if "segmentation" in name:
                if "cell-segmentation" in name:
                    table = os.path.join(ROOT, dataset, "tables", name[:-9], "default.tsv")
                    assert os.path.exists(table), table
                else:
                    table = False
                mobie.add_segmentation(path, "s0", OUT_ROOT, ds_name, im_name, resolution=[1, 1],
                                       scale_factors=[[2, 2]], chunks=(512, 512), file_format="ome.zarr",
                                       unit="pixel", max_jobs=4, tmp_folder=tmp_folder, view={},
                                       add_default_table=table)
            else:
                mobie.add_image(path, "s0", OUT_ROOT, ds_name, im_name, resolution=[1, 1],
                                scale_factors=[[2, 2]], chunks=(512, 512), file_format="ome.zarr",
                                unit="pixel", max_jobs=4, tmp_folder=tmp_folder, view={})

            mobie.metadata.set_is2d(ds_folder, True)

        sources = [["nuclei"], ["serum"], ["marker"], ["spike"], ["cell-segmentation"]]
        display_settings = [
            {"contrastLimits": [141.0, 4385.0], "visible": False, "color": "blue"},
            {"contrastLimits": [132.5, 4316.0], "visible": False, "color": "green"},
            {"contrastLimits": [72.0, 3586.0], "visible": True, "color": "red"},
            {"contrastLimits": [115.0, 2319.0], "visible": False, "color": "red"},
            {"visible": True, "showAsBoundaries": True, "boundaryThickness": 8, "showTable": True}
        ]
        mobie.create_view(ds_folder, "default", sources=sources, display_settings=display_settings,
                          display_group_names=["nuclei", "serum", "marker", "spike", "cell-segmentation"],
                          overwrite=True)


def main():
    dataset = "230107_mAB_OmScreen"
    wells = ["F03", "E02", "D03", "B02"]

    pos_per_well = 2
    for well in wells:
        copy_well(dataset.lower(), well, pos_per_well)

    dataset = "230111_Test_bindingaffinity"
    well = "E11"
    copy_well(dataset.lower(), well, pos_per_well)


if __name__ == "__main__":
    main()
