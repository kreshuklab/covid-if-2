#! /bin/bash

# Arguments:
# $1 - the plate config file, e.g. plate_configs/markers_new.json
# $2 - cuda device

config_file=$1
cuda_device=$2


echo "Process plate ${config_file} with device ${cuda_device}"

# activate the default processing env
source activate torch10
python convert_images.py ${config_file}

# activate the stardist env
source activate stardist-gpu
# CUDA_VISIBLE_DEVICES=${cuda_device} python segment_nuclei.py ${config_file}
# gpu is broken
CUDA_VISIBLE_DEVICES="" python segment_nuclei.py ${config_file}

# activate the default processing env
source activate torch10
CUDA_VISIBLE_DEVICES=${cuda_device} python segment_cells.py ${config_file}

python compute_statistics.py ${config_file}
python find_stained_cells.py ${config_file}
CUDA_VISIBLE_DEVICES=${cuda_device} python classify_cells.py ${config_file}

# python scores.py ${config_file}
