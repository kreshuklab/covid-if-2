import argparse
import os

import numpy as np
import torch
import zarr

from sklearn.metrics import accuracy_score
from torchvision.models.resnet import resnet18
from training_impl import get_loader, confusion_matrix
from tqdm import tqdm


def evaluate_model(checkpoint):
    data_path = "/scratch/pape/covid-if-2/training_data/classification/v1/test.zarr"
    with zarr.open(data_path, "r") as f:
        classes = f.attrs["classes"]
    num_classes = len(classes)

    image_shape = (152, 152)
    normalization = "minmax"

    loader = get_loader(data_path, image_shape, normalization, batch_size=256,
                        use_augmentations=False, num_workers=8, drop_last=False)
    device = torch.device("cuda")

    y_true = []
    y_pred = []

    with torch.no_grad():
        model = resnet18(num_classes=num_classes)
        model_state = torch.load(os.path.join(checkpoint, "best.pt"))["model_state"]
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

    save_path = os.path.join(checkpoint, "confusion_matrix.png")
    confusion_matrix(y_true, y_pred, classes, title=f"Accuracy: {accuracy}", save_path=save_path,
                     xticks_rotation=45)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    evaluate_model(args.checkpoint)


if __name__ == "__main__":
    main()
