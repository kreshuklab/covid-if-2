# Steps to process a plate

1. Run `convert_images.py` to convert the initial czi images into the ome.zarr data format compatible with MoBIE.
2. Run `segment_nuclei.py` to segment nuclei with stardist.
3. Run `segment_cells.py` to segment the cells based on seeded watershed from nucleus segmentation with heightmap and foreground predicted by a U-Net.
4. Run `compute_statistics.py` to measure the channel expressions per segmentated cells.
5. Run `find_stained_cells.py` to find which cells express any patterns.
6. Run `classify_cells.py` to classifiy the segmented cells into the different patterns.
7. Run `scores.py` to compute the antibody response scores for the different patterns. (TODO)

For each new plate you need to create a json file with configurations in `plate_configs`. Each of the scripts above is then called with this file as its argument.
