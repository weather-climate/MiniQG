# MiniQG

This repository accompanies our conference paper. It provides tools for computing Local Wave Activity (LWA) from two-layer quasi-geostrophic potential vorticity (QGPV) fields, detecting and classifying atmospheric blocking events, and training/evaluating an Adaptive Fourier Neural Operator (AFNO) model for QGPV field prediction.

---

## Repository Structure

```
.
├── models/
│   ├── afno.py                      # AFNO backbone architecture
│   └── losses.py                    # Loss functions and field denormalizer
├── training/
│   ├── train.py                     # Entry point of training and evaluation
│   └── trainer.py                   # Training loops, LR finder, early stopping
├── evaluation/
│   ├── metrics.py                   # Accuracy metrics
│   └── visualize.py                 # Prediction and Hovmöller visualizations
├── data/
│   └── prepare_dataset.py           # Dataset preparation pipeline
├── utils/
│   ├── lwa.py                       # LWA / FAWA computation
│   ├── compute_lwa.py               # Script to batch-compute LWA from model output
│   ├── blocking_detection.py        # Blocking event detection and classification
│   ├── run_blocking_detection.py    # Script to run blocking detection
│   ├── blocking_utils.py            # Supporting utilities for blocking analysis
│   └── data_io.py                   # NetCDF I/O helpers
└── visualization/
    ├── plot_qgpv.py                 # QGPV field snapshots
    ├── plot_blocking.py             # Blocking event trajectory plots
    └── plot_diagnostics.py          # Histograms and pixel-wise diagnostics
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

## Usage

### 1. Prepare the Dataset

Edit the configuration at the top of `data/prepare_dataset.py` to point to your raw data and set your processing config, then run:

```bash
python data/prepare_dataset.py
```

This produces a `.npz` file containing normalized training, validation, and test splits.

---

### 2. Train the AFNO Model

Open `training/train.py` and fill in the `cfg` dictionary — all fields marked `None` must be set before running:

```python
cfg = {
    'data_path':    'path/to/your_dataset.npz',
    'save_dir':     'path/to/model_output',
    'patch_size':   ...,
    'embed_dim':    ...,
    'depth':        ...,
    'batch_size':   ...,
    'epochs':       ...,
    ...
}
```

Then run:

```bash
python training/train.py
```

Training includes an optional learning rate finder, early stopping, periodic checkpointing, and an optional autoregressive fine-tuning stage controlled by `cfg['fine_tune']`.

---

### 3. Compute LWA from Model Output

Edit the `input_file` and `output_file` paths at the top of `utils/compute_lwa.py`, and set `data['predictions']` or `data['truths']` depending on whether you are processing model predictions or ground truth. Then run:

```bash
python utils/compute_lwa.py
```

This writes a NetCDF file containing `LWA_pv`, `LWA_pv_a`, and `LWA_pv_c` fields for each time step.

---

### 4. Detect Blocking Events

Edit the `input_file` and `output_file` paths in `utils/run_blocking_detection.py`, then run:

```bash
python utils/run_blocking_detection.py
```

This reads the LWA NetCDF produced in step 3 and outputs a NetCDF file containing detected blocking event trajectories, durations, peak locations, and event type classifications (ridge / trough / dipole).

---

### 5. Evaluate and Visualize

Evaluation metrics are computed automatically at the end of `training/train.py`. For standalone evaluation or visualization, import from the `evaluation/` module:

```python
from evaluation.metrics import evaluate_all_channels
from evaluation.visualize import visualize_prediction, plot_hovmoller_comparison
```

Blocking trajectories and QGPV snapshots can be plotted using the scripts in `visualization/`.

---

## Reference

The AFNO architecture is based on:

> Pathak, J., Subramanian, S., Harrington, P., Raja, S., Chattopadhyay, A., Mardani, M., Kurth, T., Hall, D., Li, Z., Azizzadenesheli, K., Hassanzadeh, P., Kashinath, K., & Anandkumar, A. (2022). FourCastNet: A global data-driven high-resolution weather model using adaptive Fourier neural operators. *arXiv preprint* arXiv:2202.11214.
