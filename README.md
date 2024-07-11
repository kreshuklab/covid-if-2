# Immunofluorescence Assay for SARS-CoV-2

This repository implements image analysis and quantification for immune response to different SARS-CoV-2 variants.
The publication describing this work will be available soon.

This work is a follow-up to [this publication](https://onlinelibrary.wiley.com/doi/full/10.1002/bies.202000257).

## Approach

The assay contains cells that express the spike proteins of different SARS-CoV-2 variants, the nucleocapsid protein and control cells not expressing any viral proteins.

The immunofluorescence data is analyzed with the following approach:
- Segmenting nuclei with StarDist.
- Predicting cell foreground and boundaries with a UNet.
- Segmenting cells with a seeded watershed based on predictions from the first two steps.
- Classifying cells into variants / ncapsid / control using a ResNet.
- Determining the antibody to the different variants based on classification results and per-cell immunofluorescence intensity.

## Installation

TODO

## Usage

TODO
