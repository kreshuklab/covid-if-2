#! /bin/bash

# Arguments:
# $1 - the plate config file, e.g. plate_configs/markers_new.json

config_file=$1


echo "Process plate ${config_file} on cpu"

# activate the default processing env
source activate torch10
python convert_images.py ${config_file}

# activate the stardist env
source activate stardist-gpu
CUDA_VISIBLE_DEVICES="" python segment_nuclei.py ${config_file}

# activate the default processing env
source activate torch10
CUDA_VISIBLE_DEVICES="" python segment_cells.py ${config_file}

python compute_intensities.py ${config_file}
CUDA_VISIBLE_DEVICES="" python classify_cells.py ${config_file}
python compute_cell_qc.py ${config_file}

python compute_scores.py ${config_file}
