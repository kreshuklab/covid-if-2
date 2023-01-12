import os
import zarr
from tqdm import tqdm

ROOT = "/scratch/pape/covid-if-2/training_data/classification"


def create_new_split(in_path, out_path, skip_classes):
    f_in = zarr.open(in_path, "r")
    classes = f_in.attrs["classes"]

    f_out = zarr.open(out_path, "a")
    if skip_classes is None:
        new_classes = classes
    else:
        new_classes = [cls for cls in classes if cls not in skip_classes]
        print(new_classes)
    f_out.attrs["classes"] = new_classes

    for k, v in f_in.attrs.items():
        if k == "classes":
            continue
        f_out.attrs[k] = v

    sample_id = 0
    for k, v in tqdm(f_in.items(), total=len(f_in)):
        cls_name = v.attrs["class_name"]
        if skip_classes is not None and cls_name in skip_classes:
            continue
        v_new = f_out.create_dataset(f"sample{sample_id:06}", data=v[:])
        for a, b in v.attrs.items():
            if a == "class_id":
                class_name = v.attrs["class_name"]
                v_new.attrs[a] = new_classes.index(class_name)
            else:
                v_new.attrs[a] = b
        sample_id += 1


def create_new_version(src, dst, skip_classes=None):
    root_in = os.path.join(ROOT, f"v{src}")
    assert os.path.exists(root_in)
    root_out = os.path.join(ROOT, f"v{dst}")
    os.makedirs(root_out, exist_ok=True)
    for split in ("train", "test", "val"):
        create_new_split(
            os.path.join(root_in, f"{split}.zarr"),
            os.path.join(root_out, f"{split}.zarr"),
            skip_classes
        )


def main():
    create_new_version(1, 5, skip_classes=["mScarlet-Lamin"])


if __name__ == "__main__":
    main()
