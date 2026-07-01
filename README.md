# Atmospheric Blocking Detection with Deep Learning

This repository accompanies our journal paper. It provides tools for computing Local Wave Activity (LWA) from two-layer quasi-geostrophic potential vorticity (QGPV) fields, detecting and classifying atmospheric blocking events, and training/evaluating an customized Adaptive Fourier Neural Operator (AFNO) model for QGPV field prediction.

---

## Repository Structure

```
.
├── models/
│   ├── afno.py                   # AFNO backbone architecture
│   └── losses.py                 # Loss functions and field denormalizer
├── training/
│   ├── train.py                  # Entry point: training and evaluation
│   └── trainer.py                # Training loops, LR finder, early stopping
├── evaluation/
│   ├── metrics.py                # Accuracy metrics (RMSE, MAE, R², etc.)
│   └── visualize.py              # Prediction and Hovmöller visualizations
├── data/
│   └── prepare_dataset.py        # Dataset preparation pipeline
├── utils/
│   ├── lwa.py                    # LWA / FAWA computation
│   ├── compute_lwa.py            # Batch-compute LWA from raw QGPV fields
│   ├── blocking_detection.py     # Blocking event detection and classification
│   ├── run_blocking_detection.py # Script to run blocking detection
│   ├── blocking_utils.py         # Supporting utilities for blocking analysis
│   └── data_io.py                # NetCDF I/O helpers
├── visualization/
│   ├── plot_qgpv.py              # QGPV field snapshots
│   ├── plot_blocking.py          # Blocking event trajectory plots
│   └── plot_diagnostics.py       # Histograms and pixel-wise diagnostics
└── submit_job.sh                 # HPC job submission script
```

---

## Requirements

```
torch
numpy
xarray
scipy
einops
matplotlib
opencv-python
psutil
netCDF4
```

Install dependencies via:

```bash
pip install torch numpy xarray scipy einops matplotlib opencv-python psutil netCDF4
```

---

## Pipeline Overview

The full workflow proceeds in five stages:

```
Raw QGPV (.nc)
    │
    ▼
[1] compute_lwa.py          →  LWA fields (.nc, 6-hourly, both layers)
    │
    ▼
[2] run_blocking_detection.py  →  Blocking events (.nc, with type labels)
    │
    ├── [3] prepare_dataset.py  →  Normalized .npz splits for training
    │         (LWA-threshold CSV used to remove high-activity days from train set)
    │
    ▼
[4] training/train.py       →  Trained AFNO model checkpoints
    │
    ▼
[5] evaluation/             →  Metrics, prediction plots, Hovmöller diagrams
```

---

## Usage

### 1. Compute LWA

Edit the `input_path` and `output_path` at the top of `utils/compute_lwa.py`, then run:

```bash
python utils/compute_lwa.py
```

This computes LWA, anticyclonic LWA, and cyclonic LWA for both QGPV layers at every time step and writes them to a NetCDF file with dimensions `(time, channel, y, x)`.

---

### 2. Detect Blocking Events

Edit the `input_file` and `output_file` paths in `utils/run_blocking_detection.py`, then run:

```bash
python utils/run_blocking_detection.py
```

This reads the LWA NetCDF from step 1 (using the `q1` channel) and outputs a NetCDF file containing detected blocking event trajectories, durations, peak locations, and type classifications (ridge / trough / dipole).

---

### 3. Prepare the Training Dataset

The dataset preparation script supports LWA-threshold-based removal of high-activity timesteps from the training set. A CSV file mapping threshold names to timestep indices (produced alongside the LWA computation) is used to identify and remove these days.

Edit the configuration at the top of `data/prepare_dataset.py`, supply the three split ratio, and run:

```bash
python data/prepare_dataset.py
```

This produces `.npz` files containing normalized training, validation, and test splits. The split boundaries are determined on the original time coordinate before any subsampling, ensuring no data leakage.

---

### 4. Train the AFNO Model

Open `training/train.py` and fill in the `cfg` dictionary — all fields set to `None` must be provided before running:

```python
cfg = {
    'data_path':         'path/to/your_dataset.npz',
    'save_dir':          'path/to/model_output',
    'train_ratio':       ...,   # e.g. 0.8
    'valid_ratio':       ...,   # e.g. 0.1
    'test_ratio':        ...,   # e.g. 0.1
    'lwa_csv_path':      'path/to/LWA_threshold_steps.csv',
    'lwa_threshold_key': ...,   # e.g. 'top_15_pct'
    'patch_size':        ...,
    'embed_dim':         ...,
    'depth':             ...,
    'batch_size':        ...,
    'epochs':            ...,
    ...
}
```

Then run:

```bash
python training/train.py
```

Training includes an optional learning rate finder, early stopping, periodic checkpointing, and an optional autoregressive fine-tuning stage.

For multi-GPU training on an HPC cluster, edit `submit_job.sh` with your environment paths and submit via:

```bash
qsub submit_job.sh
```

---

### 5. Evaluate and Visualize

Evaluation metrics are computed automatically at the end of `training/train.py`. For standalone evaluation or visualization, import directly from the `evaluation/` module:

```python
from evaluation.metrics import evaluate_all_channels
from evaluation.visualize import visualize_prediction, plot_hovmoller_comparison
```

Blocking trajectories and QGPV snapshots can be plotted using the scripts in `visualization/`.

---

## Reference

The AFNO architecture is based on:

> Guibas, J., Mardani, M., Li, Z., Tao, A., Anandkumar, A., & Catanzaro, B. (2021). Efficient Token Mixing for Transformers via Adaptive Fourier Neural Operators. In International Conference on Learning Representations.

> Pathak, J., Subramanian, S., Harrington, P., Raja, S., Chattopadhyay, A., Mardani, M., Kurth, T., Hall, D., Li, Z., Azizzadenesheli, K., Hassanzadeh, P., Kashinath, K., & Anandkumar, A. (2022). FourCastNet: A global data-driven high-resolution weather model using adaptive Fourier neural operators. *arXiv preprint* arXiv:2202.11214.