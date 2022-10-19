import argparse
import os

from elf.io import open_file
from torch_em.util.modelzoo import export_bioimageio_model, get_default_citations


SAVE_ROOT = "/scratch/pape/covid-if-2/networks/segmentation"


def export_to_bioimageio(checkpoint):

    data_path = "/scratch/pape/covid-if-2/training_data/segmentation/pseudo-labels-v1/S1.h5"
    with open_file(data_path, "r") as f:
        input_data = f["image"][:768, :768]

    name, description = os.path.basename(checkpoint), "Pseudo-label model"
    tags = ["unet", "cells", "high-content-microscopy", "instance-segmentation",
            "covid19", "immunofluorescence-microscopy", "2d"]

    # eventually we should refactor the citation logic
    # covid_if_pub = "https://doi.org/10.1002/bies.202000257"
    cite = get_default_citations(model="UNet2d", model_output="boundaries")

    doc = "# my doc"

    os.makedirs(SAVE_ROOT, exist_ok=True)
    output = os.path.join(SAVE_ROOT, name)
    export_bioimageio_model(
        checkpoint, output,
        input_data=input_data,
        name=name,
        authors=[{"name": "Constantin Pape", "affiliation": "EMBL Heidelberg"}],
        tags=tags,
        license="CC-BY-4.0",
        documentation=doc,
        description=description,
        git_repo="https://github.com/constantinpape/torch-em.git",
        cite=cite,
        input_optional_parameters=False,
        # need custom deepimagej fields if we have torchscript export
        maintainers=[{"github_user": "constantinpape"}],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    export_to_bioimageio(args.checkpoint)
