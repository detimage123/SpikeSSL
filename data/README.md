# Data

No recordings are distributed with this repository.

For training, download the public ground-truth recording sets yourself and
place them here, or pass an external directory with
`paths.data_dir=/path/to/data`. Update `configs/data/public_template.yaml` to
list the training and test files. The training loader supports CASCADE-style
`*_mini.mat` files containing `CAttached`, as well as compatible NWB files.

The browser UI does not require a dataset folder. Upload one or more
fluorescence traces directly in `.mat`, `.csv`, `.npy`, or `.npz` format.
