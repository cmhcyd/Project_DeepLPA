# Project_DeepLPA

An open-source software toolkit designed for event-level deep learning analysis
of high-energy physics data stored in the ROOT format.

## Installation

Use Python 3.9 or newer and a PyTorch build appropriate for your CPU/CUDA
environment. Install the base dependencies with:

```bash
python -m pip install -r requirements.txt
```

For Optuna search, pruning, interactive plots, and journal-file persistence:

```bash
python -m pip install -r requirements-hpo.txt
```

Uproot reads/writes ROOT files without requiring a C++ ROOT installation.
The writer explicitly produces TTrees, including when using Uproot 5.7+.
Models and losses are selected by their Python class paths and JSON arguments.

## Training

Place training ROOT files under `DATA/train/` and inference files under
`DATA/infer/`. Each TTree entry represents an event. Example (Bash syntax):

```bash
python main.py \
  --data_dir DATA --tree_name Tree1 \
  --scalar_branches mass,chisq --target_branches is_sig,is_bkg \
  --task classification --num_classes 2 --pos_class 0 \
  --model_class model.DRNmodfied.DRNmod \
  --model_args '{"width":256,"width2":64,"depth":4,"dropout":0.25}' \
  --loss_class model.FocalLoss.FocalLoss \
  --loss_args '{"signal_class":0,"alpha":0.9,"gamma":2.0}' \
  --val_ratio 0.25 --normalize --save_scaler outputs/scaler.npz \
  --early_stop_metric auc --early_stop_patience 5 \
  --epochs 100 --ckpt outputs/best.pt
```

`--array_spec '[tracks_pt,5],[tracks_eta,5]'` pads/truncates each array branch.
Array ordering is defined by the input files. Required input and target branches
must exist. Classification targets must be valid one-hot vectors or integer
class indices in `[0, num_classes)`; omit target branches for unlabeled inference.
Early-stopping metrics are `loss` and `auc`, accepting either case. Regression
uses validation loss. Batch-normalized models need training batches with more
than one event.

## Preprocessing and inference

With `--normalize`, the training/validation split is made before fitting the
mean and standard deviation on the training subset. The same statistics are
applied to validation and inference events. Save them with `--save_scaler`.
For normalized inference supply `--scaler_file outputs/scaler.npz`; this enables
normalization and takes precedence over fitting new statistics. Missing or
invalid scalers fail explicitly. Keep the feature names and ordering identical.

Use the same model configuration with `--mode infer --ckpt outputs/best.pt` and
the scaler above. For HPO, the winning model/loss configuration is recorded in
`best.parameters` in the results JSON. Models output logits; classification
probabilities are computed with Softmax during evaluation and inference.

## Outputs and metrics

The framework can save NumPy predictions and merged or per-file ROOT output,
including original branches and predictions. Probability branch numbering
remains one-based for compatibility: `pred_prob_1` corresponds to model class 0,
`pred_prob_2` to class 1, and so on. `pred_class` uses zero-based class indices.
An optional `is_train` branch records the training split.

Binary AUC treats `--pos_class` as positive. Multiclass AUC is macro-averaged
one-vs-rest. `bkg_acc` is FP/(TN+FP); `bkg_rej` is one minus that rate;
`bkg_ratio` is FP/(TP+FP), the background fraction among selected events.

FOMLoss uses `-S/(n_signal*sqrt(S+B))` when `normalize_by_pos=True`, with
numerical stabilization. All counts/sums are per mini-batch; a batch with no
signal returns a differentiable zero loss. This corrects earlier releases,
which normalized S inside the square root and did not match the stated formula.

Data loading currently keeps features in memory; batch inference does not imply
streaming ROOT IO. Integrated Gradients explains a chosen output logit from a
zero baseline and reports signed mean attribution. Diagnostic examples here
describe interfaces rather than validated physics results.

## Hyperparameter search and verification

See [HYPERPARAMETER_SEARCH.md](HYPERPARAMETER_SEARCH.md) for random, grid,
categorical TPE and Optuna search, pruning, journal files and visualization.
The optional search methods reuse one data split and initialize fresh models.
No notebook is bundled with this release.

```bash
python -m unittest discover -s tests -v
```

Tests cover synthetic ROOT preprocessing and output, the FOM formula/gradient,
early-stopping options, and hyperparameter-search behavior. Optuna/visualization
tests require their optional dependencies. These checks do not reproduce the
BESIII physics measurements or validate historical performance numbers.
