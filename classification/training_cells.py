import os
from training_impl import train_classification
from torchvision.models import resnet


def main():
    root = "/scratch/pape/covid-if-2/training_data/classification/v5"
    train_path = os.path.join(root, "train.zarr")
    val_path = os.path.join(root, "val.zarr")
    image_shape = (152, 152)

    use_mask = False
    model = "resnet18"

    name = f"classification_v5_{model}"
    if use_mask:
        name = f"{name}_with_mask"

    normalization = "minmax"
    learning_rate = 1e-4
    train_classification(name, train_path, val_path, image_shape, normalization, learning_rate,
                         use_mask=use_mask, modelCls=getattr(resnet, model))


if __name__ == "__main__":
    main()
