import argparse
import os

from czifile import CziFile


def czi_file_info(input_path):
    samples = {}
    with CziFile(input_path, "r") as f:
        for block in f.subblocks():
            sample = [d.start for d in block.dimension_entries if d.dimension == "S"][0]
            channel = [d.start for d in block.dimension_entries if d.dimension == "C"][0]
            if sample in samples:
                samples[sample].append(channel)
            else:
                samples[sample] = [channel]
    n_samples = len(samples)

    channels = None
    for this_channels in samples.values():
        if channels is None:
            channels = set(this_channels)
        else:
            assert set(this_channels) == channels, "Inconsistent channels"
    return n_samples, channels


def czi_folder_info(folder):
    for root, dirs, files in os.walk(folder):
        for ff in files:
            assert ff.endswith(".czi")
            path = os.path.join(root, ff)
            n_samples, channels = czi_file_info(path)
            print(path, ": channels:", channels, "samples:", n_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()
    czi_folder_info(args.folder)


if __name__ == "__main__":
    main()
