import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision.models import resnet
from training_impl import get_loader
from tqdm import tqdm


def get_class_names(model_version):
    if model_version == 1:
        return ["3xNLS-mScarlet", "LCK-mScarlet", "mScarlet-H2A", "mScarlet-Lamin", "mScarlet-Giantin"]
    elif model_version == 2:
        return ["3xNLS-mScarlet", "LCK-mScarlet", "mScarlet-H2A", "mScarlet-Lamin", "mScarlet-Giantin", "untagged"]
    elif model_version in (3, 5):
        return ["3xNLS-mScarlet", "LCK-mScarlet", "mScarlet-H2A", "mScarlet-Giantin", "mScarlet-Lamin"]
    elif model_version == 4:
        return ["mScarlet-Giantin", "LCK-mScarlet", "mScarlet-H2A", "3xNLS-mScarlet", "mScarlet-Lamin"]
    else:
        raise ValueError(f"Invalid model version {model_version}")


def evaluate_model(checkpoint, use_mask, class_names, data_version):
    data_path = f"/scratch/pape/covid-if-2/training_data/manual/v{data_version}/data.zarr"
    num_classes = len(class_names)

    image_shape = (152, 152)
    normalization = "minmax"

    loader = get_loader(data_path, image_shape, normalization, batch_size=256,
                        use_augmentations=False, num_workers=8, drop_last=False,
                        use_mask=use_mask, class_names=class_names)
    device = torch.device("cuda")

    y_true = []
    y_pred = []

    with torch.no_grad():
        if "resnet" in checkpoint:
            if use_mask:
                model_name = checkpoint.split("_")[-3]
            else:
                model_name = checkpoint.split("_")[-1]
            model = getattr(resnet, model_name)(num_classes=num_classes)
        else:
            model = resnet.resnet18(num_classes=num_classes)
        model_state = torch.load(os.path.join(checkpoint, "best.pt"))["model_state"]
        # save the state dict so we can more easily reload it
        torch.save(model_state, os.path.join(checkpoint, "best_model.pt"))
        model.load_state_dict(model_state)
        model.eval()
        model.to(device)
        for x, y in tqdm(loader):
            pred = model(x.to(device))
            y_true.append(y.numpy())
            y_pred.append(pred.max(1)[1].cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)

    labels = [cname.replace("mScarlet-", "").replace("-mScarlet", "") for cname in class_names]
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    tab = pd.DataFrame(cm, columns=labels, index=labels)
    print(tab.to_markdown())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--model_version", required=True, type=int)
    args = parser.parse_args()

    model_version = args.model_version
    data_version = 1

    class_names = get_class_names(model_version)

    checkpoints = glob(os.path.join(f"checkpoints/classification_v{model_version}*"))
    for ckpt in checkpoints:
        print(ckpt)
        use_mask = "with_mask" in ckpt
        evaluate_model(ckpt, use_mask, class_names, data_version)


if __name__ == "__main__":
    main()
