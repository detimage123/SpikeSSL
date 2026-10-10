# SpikeSSL

**SpikeSSL** is a universal framework for inferring neural spiking activity
from calcium-imaging fluorescence traces. Its dynamics-informed backbone uses
bidirectional IIR state-space layers, multimodal conditioning, a sparse gate,
and a heteroscedastic variance head to support both in-domain and
unseen-indicator inference.

This repository contains the released implementation, a browser-based
inference GUI, and the single checkpoint used by that GUI.

## Contents and release boundary

Included:

- Core SpikeSSL model, training code, and a public-data configuration template.
- A local browser UI for inference from uploaded fluorescence traces.
- `checkpoints/spikessl_ui_default.ckpt`, the only released checkpoint.

Not included:

- Private raw or reconstructed recordings, private evaluation outputs, and
  cached GUI results.
- Simulation data. Simulation assets and their release instructions will be
  handled separately.
- Public benchmark recordings. Obtain them from their original sources before
  training or benchmarking.

## Installation

Python 3.10 and an NVIDIA GPU are recommended. Install a PyTorch build that
matches your CUDA installation, then install the remaining dependencies:

```bash
conda create -n spikessl python=3.10
conda activate spikessl
pip install -r requirements.txt
```

For a CUDA 12.1 Conda environment, `environment.yml` provides a tested setup:

```bash
conda env create -f environment.yml
conda activate spikessl-webui
```

## Run the GUI

```bash
bash start_gui.sh
```

Then open `http://127.0.0.1:7860` locally, or
`http://<server-ip>:7860` from another machine. Set `CUDA_VISIBLE_DEVICES`,
`SPIKESSL_HOST`, or `SPIKESSL_PORT` when needed:

```bash
CUDA_VISIBLE_DEVICES=1 SPIKESSL_PORT=7861 bash start_gui.sh
```

The GUI accepts `.mat`, `.csv`, `.npy`, and `.npz` files. A 1D array is treated
as one trace; for 2D arrays the expected layout is `[traces, frames]`, and an
obvious `[frames, traces]` layout is transposed automatically. Uploaded inputs
and temporary converted files are removed after inference unless the user
explicitly selects **Keep Uploaded Input Data On Server**.

## Train on public ground-truth recordings

The template expects user-provided recordings under `data/`. Edit
`configs/data/public_template.yaml` to list your training and test files, then
run:

```bash
PROJECT_ROOT="$PWD" python src/train.py
```

An external dataset root can be used without copying recordings into the
repository:

```bash
PROJECT_ROOT="$PWD" python src/train.py paths.data_dir=/path/to/data
```

## Repository layout

```text
checkpoints/       Released GUI checkpoint only
configs/           Training and model configuration
data/              Empty local mount point for user-provided public data
scripts/           GUI inference entry point and shared inference utilities
src/               Model, data loading, training, and utility code
webui/             Local browser interface
result/            Runtime outputs, ignored by Git
```

## Citation

If you use SpikeSSL, please cite our paper:

```bibtex
@article{yue2026spikessl,
  title={SpikeSSL: A Universal Spike Inference Framework with Dynamics-Informed State-Space Layers},
  author={Yue, Chenghao and Xing, Siming and Liu, Shuran and Li, Angran and Zhang, Yuanlong},
  journal={arXiv preprint arXiv:2610.11456},
  year={2026},
  url={https://arxiv.org/abs/2610.11456}
}
```

Before publishing the repository, add the license selected by the authors.
