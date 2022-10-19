import os
from training_impl import train_classification


def main():
    root = "/scratch/pape/covid-if-2/training_data/cifar"
    train_path = os.path.join(root, "train.zarr")
    val_path = os.path.join(root, "val.zarr")
    image_shape = (168, 168)
    name = "cifar"
    normalization = "minmax"
    learning_rate = 1e-4
    train_classification(name, train_path, val_path, image_shape, normalization, learning_rate)


if __name__ == "__main__":
    main()
