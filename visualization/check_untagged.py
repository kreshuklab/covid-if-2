import napari
from czifile import CziFile


def check_untagged(path):
    images = []
    with CziFile(path, "r") as f:
        for block in f.subblocks():
            im = block.data().squeeze()
            images.append(im)

    v = napari.Viewer()
    for im in images:
        v.add_image(im)
    napari.run()


if __name__ == "__main__":
    path = "/g/kreshuk/data/covid-if-2/from_nuno/20221210_mSc_untagged/mScarlet_untagged_fixed-Scene-1-P1-A03.czi"
    check_untagged(path)
