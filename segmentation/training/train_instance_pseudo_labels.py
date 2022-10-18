import os
from glob import glob

import torch
import torch_em
from torch_em.model import UNet2d

ROOT = ""


def get_loader(split, patch_shape, batch_size):
    file_paths = glob(os.path.join(ROOT, "*.h5"))
    file_paths.sort()
    val_paths = file_paths[::8]
    if split == "train":
        file_paths = list(set(file_paths) - set(val_paths))
        n_samples = 1000
    else:
        file_paths = val_paths
        n_samples = 50

    label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True)

    raw_key = "image"
    label_key = "labels"
    return torch_em.default_segmentation_loader(
        file_paths, raw_key,
        file_paths, label_key,
        patch_shape=patch_shape, batch_size=batch_size,
        label_transform=label_transform,
        num_workers=8,
        n_samples=n_samples,
    )


def train_boundaries(args):
    model = UNet2d(in_channels=1, out_channels=2, initial_features=64, final_activation="Sigmoid")

    patch_shape = (768, 768)
    train_loader = get_loader("train", patch_shape, args.batch_size)
    val_loader = get_loader("test", patch_shape, args.batch_size)
    loss = torch_em.loss.DiceLoss()

    trainer = torch_em.default_segmentation_trainer(
        name="instance-pseudo-labels",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        device=torch.device("cuda"),
        mixed_precision=True,
        log_image_interval=50
    )
    trainer.fit(iterations=args.n_iterations)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper(default_batch_size=8)
    args = parser.parse_args()
    train_boundaries(args)
