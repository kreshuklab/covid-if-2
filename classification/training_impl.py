#
# A lot of this should eventually be refactored into torch_em to enable classification there.
#

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_em

from elf.io import open_file
from matplotlib.backends.backend_agg import FigureCanvasAgg
from skimage.transform import resize
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from torch_em.trainer.logger_base import TorchEmLogger
from torchvision.models.resnet import resnet18
# from torchvision.utils import make_grid


class ClassificationTrainer(torch_em.trainer.DefaultTrainer):
    def _validate_impl(self, forward_context):
        self.model.eval()

        loss_val = 0.0

        # we use the syntax from sklearn.metrics to compute metrics
        # over all the preditions
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                with forward_context():
                    pred, loss = self._forward_and_loss(x, y)
                loss_val += loss.item()
                y_true.append(y.detach().cpu().numpy())
                y_pred.append(pred.max(1)[1].detach().cpu().numpy())

        if torch.isnan(pred).any():
            print("Nannnnnnnnnnnnnnnnnnnnnnnaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaannnnnnnn")
        loss_val /= len(self.val_loader)

        y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
        metric_val = self.metric(y_true, y_pred)

        if self.logger is not None:
            self.logger.log_validation(self._iteration, metric_val, loss_val, x, y, pred, y_true, y_pred)
        return metric_val


def confusion_matrix(y_true, y_pred, class_labels=None, title=None, save_path=None, **plot_kwargs):
    fig, ax = plt.subplots(1)

    if save_path is None:
        canvas = FigureCanvasAgg(fig)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, normalize="true", display_labels=class_labels
    )
    disp.plot(ax=ax, **plot_kwargs)

    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        plt.savefig(save_path)
        return

    canvas.draw()
    image = np.asarray(canvas.buffer_rgba())[..., :3]
    image = image.transpose((2, 0, 1))
    plt.close()
    return image


# TODO normalization and stuff
# TODO get the class names
def make_grid(images, target=None, prediction=None, images_per_row=8, **kwargs):
    assert images.ndim == 4
    assert images.shape[1] in (1, 3)

    n_images = images.shape[0]
    n_rows = n_images // images_per_row
    if n_images % images_per_row != 0:
        n_rows += 1

    images = images.detach().cpu().numpy()
    if target is not None:
        target = target.detach().cpu().numpy()
    if prediction is not None:
        prediction = prediction.max(1)[1].detach().cpu().numpy()

    fig, axes = plt.subplots(n_rows, images_per_row)
    canvas = FigureCanvasAgg(fig)
    for r in range(n_rows):
        for c in range(images_per_row):
            i = r * images_per_row + c
            ax = axes[r, c]
            ax.set_axis_off()
            im = images[i]
            im = im.transpose((1, 2, 0))
            if im.shape[-1] == 3:  # rgb
                ax.imshow(im)
            else:
                ax.imshow(im[..., 0], cmap="gray")

            if target is None and prediction is None:
                continue

            # TODO get the class name, and if we have both target
            # and prediction check whether they agree or not and do stuff
            title = ""
            if target is not None:
                title += f"t: {target[i]} "
            if prediction is not None:
                title += f"p: {prediction[i]}"
            ax.set_title(title, fontsize=8)

    canvas.draw()
    image = np.asarray(canvas.buffer_rgba())[..., :3]
    image = image.transpose((2, 0, 1))
    plt.close()
    return image


class ClassificationLogger(TorchEmLogger):
    def __init__(self, trainer, save_root, **unused_kwargs):
        super().__init__(trainer, save_root)
        self.log_dir = f"./logs/{trainer.name}" if save_root is None else\
            os.path.join(save_root, "logs", trainer.name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.tb = torch.utils.tensorboard.SummaryWriter(self.log_dir)
        self.log_image_interval = trainer.log_image_interval

    def add_image(self, x, y, pred, name, step):
        scale_each = False
        marker = make_grid(x[:, 0:1], y, pred, padding=4, normalize=True, scale_each=scale_each)
        self.tb.add_image(tag=f"{name}/marker", img_tensor=marker, global_step=step)
        nucleus = make_grid(x[:, 1:2], padding=4, normalize=True, scale_each=scale_each)
        self.tb.add_image(tag=f"{name}/nucleus", img_tensor=nucleus, global_step=step)
        mask = make_grid(x[:, 2:], padding=4)
        self.tb.add_image(tag=f"{name}/mask", img_tensor=mask, global_step=step)

    def log_train(self, step, loss, lr, x, y, prediction, log_gradients=False):
        self.tb.add_scalar(tag="train/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="train/learning_rate", scalar_value=lr, global_step=step)
        if step % self.log_image_interval == 0:
            self.add_image(x, y, prediction, "train", step)

    def log_validation(self, step, metric, loss, x, y, prediction, y_true=None, y_pred=None):
        self.tb.add_scalar(tag="validation/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="validation/metric", scalar_value=metric, global_step=step)
        self.add_image(x, y, prediction, "validation", step)
        if y_true is not None and y_pred is not None:
            cm = confusion_matrix(y_true, y_pred)
            self.tb.add_image(tag="validation/confusion_matrix", img_tensor=cm, global_step=step)


class CellClassificationDataset(torch.utils.data.Dataset):
    normalizations = ("minmax", "independent_percentile", "percentile", "median")
    eps = 1e-7

    def _normalize_percentile(self, x, stats):
        assert x.shape[0] == len(self.channels), f"{x.shape}, {len(self.channels)}"
        out = []
        for channel_name, channel in zip(self.channels, x):
            # mask is already normalized
            if channel_name == "mask":
                out.append(channel.astype("float32")[None])
                continue
            p_dn, p_up = stats[f"{channel_name}_q05"], stats[f"{channel_name}_q95"]
            channel = (channel.astype("float32") - p_dn) / (p_up + self.eps)
            out.append(channel[None])
        return np.concatenate(out, axis=0)

    def _normalize_percentile_independent(self, x, stats):
        assert x.shape[0] == len(self.channels), f"{x.shape}, {len(self.channels)}"
        out = []
        for channel_name, channel in zip(self.channels, x):
            # mask is already normalized
            if channel_name == "mask":
                out.append(channel.astype("float32")[None])
                continue
            p_dn, p_up = np.percentile(channel, 4), np.percentile(channel, 96)
            channel = (channel.astype("float32") - p_dn) / (p_up + self.eps)
            out.append(channel[None])
        return np.concatenate(out, axis=0)

    # can be vectorized....
    def _normalize_minmax(self, x, stats):
        assert x.shape[0] == len(self.channels), f"{x.shape}, {len(self.channels)}"
        out = []
        for channel_name, channel in zip(self.channels, x):
            p_dn, p_up = channel.min(), channel.max()
            channel = (channel.astype("float32") - p_dn) / (p_up - p_dn + self.eps)
            out.append(channel[None])
        return np.concatenate(out, axis=0)

    # TODO
    def _normalize_median(self, x, stats):
        raise NotImplementedError

    def __init__(self, data_path, image_shape, normalization,
                 augmentation=None, use_mask=False, class_names=None):
        self.f = open_file(data_path, "r")

        self.class_names = class_names
        if self.class_names is None:
            self.n_classes = len(self.f.attrs["classes"])
        else:
            self.n_classes = len(self.class_names)

        self.channels = self.f.attrs["channels"]
        self.n_samples = len([k for k in self.f if k.startswith("sample")])
        self.augmentation = augmentation
        assert len(image_shape) == 2, f"{image_shape}"
        self.image_shape = image_shape

        assert normalization in self.normalizations
        if normalization == "percentile":
            self.normalization = self._normalize_percentile
        elif normalization == "minmax":
            self.normalization = self._normalize_minmax
        elif normalization == "independent_percentile":
            self.normalization = self._normalize_percentile_independent
        else:
            self.normalization = self._normalize_median

        self.use_mask = use_mask

    def __len__(self):
        return self.n_samples

    def resize(self, x):
        out = [
            resize(channel, self.image_shape, preserve_range=True)[None] for channel in x
        ]
        return np.concatenate(out, axis=0)

    def apply_mask(self, x):
        mask = x[-1].astype("bool")
        x[0][~mask] = 0
        x[1][~mask] = 0
        return x

    def __getitem__(self, index):
        key = f"sample{index:06}"
        ds = self.f[key]

        # process the target
        if self.class_names is None:
            y = ds.attrs["class_id"]
        else:
            y = self.class_names.index(ds.attrs["class_name"])
        assert 0 <= y < self.n_classes

        # process the data/input:
        x = ds[:]
        # apply normalization
        x = self.normalization(x, ds.attrs.get("statistics", None))
        if self.use_mask:
            x = self.apply_mask(x)
        # resize to sample shape
        x = self.resize(x)
        # apply augmentations (if any)
        if self.augmentation:
            _shape = x.shape
            # FIXME adds unwanted batch axis
            x = self.augmentation(x)[0][0]
            assert x.shape == _shape
        return x, y


def get_loader(data_path, image_shape, normalization, batch_size,
               use_augmentations=True, num_workers=8, drop_last=True,
               use_mask=False, class_names=None):
    if use_augmentations:
        augmentation = torch_em.transform.get_augmentations(ndim=2)
    else:
        augmentation = None
    ds = CellClassificationDataset(data_path, image_shape, normalization, augmentation=augmentation,
                                   use_mask=use_mask, class_names=class_names)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=drop_last
    )
    loader.shuffle = True
    return loader


def get_num_classes(path):
    with open_file(path, "r") as f:
        n_classes = len(f.attrs["classes"])
    return n_classes


def train_classification(
    name, train_path, val_path, image_shape, normalization, learning_rate,
    batch_size=32, use_mask=False, modelCls=None,
):
    num_classes = get_num_classes(train_path)
    print("Start training with", num_classes, "classes")

    train_loader = get_loader(train_path, image_shape, normalization, batch_size, use_mask=use_mask)
    val_loader = get_loader(val_path, image_shape, normalization, batch_size, use_mask=use_mask)

    if modelCls is None:
        model = resnet18(num_classes=num_classes)
    else:
        model = modelCls(num_classes=num_classes)
    loss = torch.nn.CrossEntropyLoss()

    # metric: note that we use lower metric = better !
    # so we record the accuracy error instead of the error rate
    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader,
        loss=loss, metric=lambda a, b: 1 - accuracy_score(a, b),
        learning_rate=learning_rate,
        logger=ClassificationLogger,
        trainer_class=ClassificationTrainer,
    )
    trainer.fit(int(5e4))
