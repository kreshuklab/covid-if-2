import os

import h5py
from czifile import CziFile
from tqdm import tqdm


def parse_samples(input_path):
    samples = {}
    with CziFile(input_path, "r") as f:
        for block in f.subblocks():
            sample = [d.start for d in block.dimension_entries if d.dimension == "S"][0]
            x = [d.start for d in block.dimension_entries if d.dimension == "X"][0]
            y = [d.start for d in block.dimension_entries if d.dimension == "Y"][0]
            samples[sample] = {"x": x, "y": y}
    return samples


def import_single_images(input_path, output_folder):
    channel_names = ["marker", "serum", "nuclei"]
    samples = parse_samples(input_path)
    os.makedirs(output_folder, exist_ok=True)
    with CziFile(input_path, "r") as f:
        for block in tqdm(f.subblocks()):
            sample = [d.start for d in block.dimension_entries if d.dimension == "S"][0]
            channel = [d.start for d in block.dimension_entries if d.dimension == "C"][0]
            im = block.data().squeeze()
            out_path = os.path.join(output_folder, f"sample{sample:03}.h5")
            with h5py.File(out_path, "a") as f:
                f.attrs["x"] = samples[sample]["x"]
                f.attrs["y"] = samples[sample]["y"]
                f.create_dataset(channel_names[channel], data=im, compression="gzip")


if __name__ == "__main__":
    path = "/home/pape/Work/data/covid-if2/96well_ANA_Serum_SN_dilutions-Split-01.czi"
    output_folder = "/home/pape/Work/data/covid-if2/processed-v1"
    import_single_images(path, output_folder)
