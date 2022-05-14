import os

import h5py
from czifile import CziFile
from tqdm import tqdm


def import_single_images(input_path, output_folder):
    images = {}
    os.makedirs(output_folder, exist_ok=True)

    with CziFile(input_path, "r") as f:
        for x in f.subblocks():
            sample = [d.start for d in x.dimension_entries if d.dimension == "S"][0]
            channel = [d.start for d in x.dimension_entries if d.dimension == "C"][0]

            im = x.data().squeeze()
            if sample in images:
                images[sample][channel] = im
            else:
                images[sample] = {channel: im}

    names = ["marker", "serum", "nuclei"]
    for sample, channels in tqdm(images.items(), total=len(images)):
        out_path = os.path.join(output_folder, f"data-s{sample:03}.h5")
        with h5py.File(out_path, "w") as f:
            for chan, name in enumerate(names):
                f.create_dataset(name, data=channels[chan], compression="gzip")


if __name__ == "__main__":
    path = "/home/pape/Work/data/covid-if2/96well_ANA_Serum_SN_dilutions-Split-01.czi"
    output_folder = "/home/pape/Work/data/covid-if2/test-data"
    import_single_images(path, output_folder)
