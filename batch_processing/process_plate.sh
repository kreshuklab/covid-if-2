#! /bin/bash

# Arguments:
# $1 - the plate config file, e.g. plate_configs/markers_new.json
# $2 - cuda device
# $3 - selected position (optional)

config_file=$1
cuda_device=$2
selected_position=${3:-none}


echo "Process plate ${config_file} with device ${cuda_device}"

# activate the default processing env
source activate torch10
if [[ ${selected_position} == "none" ]]
then
    python convert_images.py ${config_file}
else
    python convert_images.py ${config_file} --position ${selected_position}
fi

# activate the stardist env
source activate stardist-gpu
# CUDA_VISIBLE_DEVICES=${cuda_device} python segment_nuclei.py ${config_file}
# gpu is broken
CUDA_VISIBLE_DEVICES="" python segment_nuclei.py ${config_file}

# activate the default processing env
source activate torch10
# CUDA_VISIBLE_DEVICES=${cuda_device} python segment_cells.py ${config_file}
python segment_cells.py ${config_file}

python compute_intensities.py ${config_file}

# CUDA_VISIBLE_DEVICES=${cuda_device} python classify_cells.py ${config_file}
python classify_cells.py ${config_file}

python compute_cell_qc.py ${config_file}

python compute_scores.py ${config_file}
