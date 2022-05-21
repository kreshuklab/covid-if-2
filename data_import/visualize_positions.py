import z5py
import napari
import numpy as np
from czifile import CziFile
from pybdv.downsample import downsample
from tqdm import tqdm

TMP_PATH = "tmp.n5"


def create_overview_image(input_path):
    samples = {}
    xs, ys = [], []
    xlen, ylen = None, None
    with CziFile(input_path, "r") as f:
        for block in f.subblocks():
            sample = [d.start for d in block.dimension_entries if d.dimension == "S"][0]
            c = [d.start for d in block.dimension_entries if d.dimension == "C"][0]
            x = [d.start for d in block.dimension_entries if d.dimension == "X"][0]
            y = [d.start for d in block.dimension_entries if d.dimension == "Y"][0]

            if xlen is None:
                xlen = [d.size for d in block.dimension_entries if d.dimension == "X"][0]
            else:
                assert xlen == [d.size for d in block.dimension_entries if d.dimension == "X"][0]
            if ylen is None:
                ylen = [d.size for d in block.dimension_entries if d.dimension == "Y"][0]
            else:
                assert ylen == [d.size for d in block.dimension_entries if d.dimension == "Y"][0]

            samples[sample] = {"c": c, "x": x, "y": y}
            xs.append(x)
            ys.append(y)

    shape = (max(ys) + 1, max(xs) + 1)
    print(shape)

    with z5py.File(TMP_PATH, "w") as fout:
        ds = fout.create_dataset("data/s0", shape=shape, compression="gzip", dtype="uint8", chunks=(1024, 1024))
        ds.n_threads = 8
        with CziFile(input_path, "r") as f:
            for block in tqdm(f.subblocks()):
                sample = [d.start for d in block.dimension_entries if d.dimension == "S"][0]
                pos = samples[sample]
                x, y = pos["x"], pos["y"]
                bb = np.s_[y:y+ylen, x:x+xlen]
                ds[bb] = (sample + 1)


def downsample_overview_image(path):
    scale_factor = (4, 4)
    for scale in range(1, 4):
        print("Downsample scale", scale)
        in_key = f"data/s{scale-1}"
        out_key = f"data/s{scale}"
        downsample(path, in_key, out_key, scale_factor, n_threads=8, mode="nearest")


def main():
    path = "/home/pape/Work/data/covid-if2/96well_ANA_Serum_SN_dilutions-Split-01.czi"
    create_overview_image(path)
    downsample_overview_image(TMP_PATH)
    # TODO downsample it and load downsampled version
    with z5py.File(TMP_PATH, "r") as f:
        ds = f["data/s2"]
        ds.n_threads = 8
        v = napari.Viewer()
        v.add_labels(ds, name="positions")
        napari.run()


main()
