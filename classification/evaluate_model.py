import argparse
import os
from glob import glob

import numpy as np
import torch
import zarr

from sklearn.metrics import accuracy_score
from torchvision.models import resnet
from training_impl import get_loader, confusion_matrix
from tqdm import tqdm


def evaluate_model(checkpoint, use_mask, version=3):
    data_path = f"/scratch/pape/covid-if-2/training_data/classification/v{version}/test.zarr"
    with zarr.open(data_path, "r") as f:
        classes = f.attrs["classes"]
    num_classes = len(classes)

    image_shape = (152, 152)
    normalization = "minmax"

    loader = get_loader(data_path, image_shape, normalization, batch_size=256,
                        use_augmentations=False, num_workers=8, drop_last=False,
                        use_mask=use_mask)
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

    # save_path = os.path.join(checkpoint, "confusion_matrix.png")
    # confusion_matrix(y_true, y_pred, classes, title=f"Accuracy: {accuracy}", save_path=save_path,
    #                  xticks_rotation=45)


def evaluate_version(version):
    checkpoints = glob(os.path.join(f"checkpoints/classification_v{version}*"))
    for ckpt in checkpoints:
        use_mask = "with_mask" in ckpt
        print("Evaluating", ckpt, "with mask:", use_mask)
        evaluate_model(ckpt, use_mask, version=version)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-checkpoint", "--c", default=None)
    parser.add_argument("--use-mask", "-m", default=0, type=int)
    parser.add_argument("-v", "--version")
    args = parser.parse_args()
    if args.version is None:
        evaluate_model(args.checkpoint, bool(args.use_mask))
    else:
        evaluate_version(args.version)


if __name__ == "__main__":
    main()
