# TODO refactor into torch_em check_loader
import os
import numpy as np
from elf.visualisation import simple_grid_view
from training_impl import get_loader


def visualize_batch(x, y):
    images = [xx[0] for xx in x.numpy()]
    image_data = {"images": images}
    labels = [np.full(images[0].shape, i + 1) for i in y.numpy()]
    label_data = {"class-labels": labels}
    simple_grid_view(image_data, label_data)


def check_loader(data_path, image_shape, normalization, batch_size, n_batches):
    loader = get_loader(data_path, image_shape, normalization, batch_size, num_workers=2)
    for i, (x, y) in enumerate(loader):
        if i == n_batches:
            break
        visualize_batch(x, y)


def main():
    root = "/scratch/pape/covid-if-2/training_data/classification/v1"
    train_path = os.path.join(root, "train.zarr")
    image_shape = (152, 152)
    normalization = "minmax"
    check_loader(train_path, image_shape, normalization, batch_size=36, n_batches=4)


if __name__ == "__main__":
    main()
