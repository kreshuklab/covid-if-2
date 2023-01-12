import os
import numpy as np
from training_impl import get_loader


def check_dataset(version):
    root = f"/scratch/pape/covid-if-2/training_data/classification/v{version}"
    train_path = os.path.join(root, "train.zarr")

    image_shape = (152, 152)
    use_mask = False

    normalization = "minmax"
    batch_size = 256
    train_loader = get_loader(train_path, image_shape, normalization, batch_size, use_mask=use_mask,
                              drop_last=False)

    labels = []
    for x, y in train_loader:
        labels.extend(y.numpy().squeeze().tolist())

    label_ids, label_counts = np.unique(labels, return_counts=True)
    for lid, c in zip(label_ids, label_counts):
        print(lid, ":", c)


if __name__ == "__main__":
    check_dataset(5)
