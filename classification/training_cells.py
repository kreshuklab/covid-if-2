import os
from training_impl import train_classification


# TODO:
# - improve cell segmentation (cellpose nucleus model?) -> wait for mobie working
# - filtering for segmentation errors (based on hard edges) -> wait for mobie working
# - visualisation: display class and prediction labels on top of grid
# - add augmentations
def main():
    root = "/scratch/pape/covid-if-2/training_data/classification/v1"
    train_path = os.path.join(root, "train.zarr")
    val_path = os.path.join(root, "val.zarr")
    image_shape = (152, 152)
    # name = "classification_v1_augmentations"
    name = "classification_v1_x"
    normalization = "minmax"
    learning_rate = 1e-4
    train_classification(name, train_path, val_path, image_shape, normalization, learning_rate)


if __name__ == "__main__":
    main()
