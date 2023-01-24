import os
from training_impl import train_classification
from torchvision.models import resnet


def main():
    # root = "/scratch/pape/covid-if-2/training_data/classification/v5"
    root = "/scratch/pape/covid-if-2/training_data/manual/v1"
    train_path = os.path.join(root, "train.zarr")
    val_path = os.path.join(root, "val.zarr")
    image_shape = (152, 152)

    use_mask = True
    model = "resnet34"

    name = f"classification_manual_v1_{model}_pretrained"
    if use_mask:
        name = f"{name}_with_mask"

    pretrain_ckpt = "checkpoints/classification_v3_resnet34_with_mask/best.pt"

    normalization = "minmax"
    learning_rate = 1e-4
    train_classification(name, train_path, val_path, image_shape, normalization, learning_rate,
                         use_mask=use_mask, modelCls=getattr(resnet, model), pretrain_ckpt=pretrain_ckpt)


if __name__ == "__main__":
    main()
