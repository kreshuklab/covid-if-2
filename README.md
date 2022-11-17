# covid-if-2

## Classification Results

For confusion matrices see 'results/cm'

Accuracies:
- v1 (resnet18)
    - vanilla: 0.9412
    - augmentations: 0.9592


## Approach

- Segmentation
    - first round: run segmentation with pretrained net from covid-if-1
    - second round: train new network using first results as pseudo-labels
- Classification
    - Cut out marker, nucleus staining and object mask
    - Use first six images per well for training, image 7 for validation, image 8 + 9 for testing
    - Train a ResNet18 with aughmentations
