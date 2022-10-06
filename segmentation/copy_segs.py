from glob import glob
import h5py

inputs = glob("/home/pape/Work/data/covid-if2/test-data-processed/*.h5")
inputs.sort()

outputs = glob("/home/pape/Work/data/covid-if2/processed-v1/*.h5")
outputs.sort()


cell_key = "segmentation/cells"
nuc_key = "segmentation/nuclei"
for inp, outp in zip(inputs, outputs):
    with h5py.File(inp, "r") as f:
        cells = f[cell_key][:]
        nuclei = f[nuc_key][:]
    with h5py.File(outp, "a") as f:
        f.create_dataset(cell_key, data=cells, compression="gzip")
        f.create_dataset(nuc_key, data=nuclei, compression="gzip")
