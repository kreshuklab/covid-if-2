import os
from glob import glob

import imageio
from elf.io import open_file
from tqdm import tqdm


OUTPUT_ROOT = "/scratch/pape/covid-if-2/training_data/cifar"
INPUT_PATH = "/scratch/pape/covid-if-2/training_data/cifar10-data"


def prepare_training_data():
    train_root = os.path.join(INPUT_PATH, "train")
    test_root = os.path.join(INPUT_PATH, "test")
    roots = {"train": train_root, "test": test_root}

    classes = os.listdir(train_root)
    channels = ["red", "green", "blue"]

    out_root = OUTPUT_ROOT
    os.makedirs(out_root, exist_ok=True)

    f_train = open_file(os.path.join(out_root, "train.zarr"), "a")
    f_train.attrs["classes"] = classes
    f_train.attrs["channels"] = channels

    f_val = open_file(os.path.join(out_root, "val.zarr"), "a")
    f_val.attrs["classes"] = classes
    f_val.attrs["channels"] = channels

    f_test = open_file(os.path.join(out_root, "test.zarr"), "a")
    f_test.attrs["classes"] = classes
    f_test.attrs["channels"] = channels

    train_id, val_id, test_id = 0, 0, 0
    val_fraction = 0.1

    for split, root in roots.items():
        for class_name in tqdm(os.listdir(root), desc=f"Convert {split}"):

            class_id = classes.index(class_name)
            files = glob(os.path.join(root, class_name, "*.png"))
            n_images = len(files)
            val_switch = n_images - int(val_fraction * n_images)

            if split == "train":
                f = f_train
                sample_id = train_id
            else:
                f = f_test
                sample_id = test_id

            for i, ff in enumerate(files):
                data = imageio.imread(ff)
                data = data.transpose((2, 0, 1))
                assert data.shape[0] == 3

                if split == "train" and i == val_switch:
                    train_id = sample_id
                    sample_id = val_id
                    f = f_val

                ds = f.create_dataset(f"sample{sample_id:06}", data=data, chunks=data.shape)
                ds.attrs["class_id"] = class_id
                ds.attrs["statistics"] = {}
                sample_id += 1

            if split == "train":
                val_id = sample_id
            else:
                test_id = sample_id

    print("Number of training samples  :", train_id + 1)
    print("Number of validation samples:", val_id + 1)
    print("Number of test samples      :", test_id + 1)


def main():
    prepare_training_data()


if __name__ == "__main__":
    main()
