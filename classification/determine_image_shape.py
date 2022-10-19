import numpy as np
from elf.io import open_file
from tqdm import tqdm


def determine_image_shape():
    path = "/scratch/pape/covid-if-2/training_data/classification/v1/train.zarr"
    shapes = []
    with open_file(path, "r") as f:
        for ds in tqdm(f.values()):
            shape = ds.shape
            assert shape[0] == 3
            shapes.append(shape[1:])
    print(np.mean(shapes, axis=0))


determine_image_shape()
