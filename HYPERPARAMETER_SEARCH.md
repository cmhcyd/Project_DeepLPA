# Automatic hyperparameter search

Add `--hpo` to an existing training command. Search loads the data once and uses
the same training/validation split for every candidate. It trains fresh models and saves the best
validation checkpoint to `--ckpt`.

If normalization is enabled, statistics are fitted only on the training subset
before any trial starts, and shared with the validation/inference data. Save the
statistics with `--save_scaler` and reuse them with `--scaler_file` for inference.
See [README.md](README.md) for installation, label validation, and preprocessing.

## Search methods

Select a method with `--hpo_method`:

| Method | Behavior | Suitable use |
| --- | --- | --- |
| `random` (default) | Samples combinations uniformly without replacement | Baseline exploration |
| `grid` | Visits combinations in JSON parameter/value order | Small spaces that can be fully evaluated |
| `tpe` | Uses previous validation scores to select later combinations | Larger spaces with a limited trial budget |
| `optuna` | Uses Optuna's official TPE sampler | Search with the maintained Optuna implementation |

### Optuna

Install the optional dependency in the Python environment used for training:

```bash
python -m pip install 'optuna>=4'
```

Add these options to an existing training command (including its data settings):

```bash
--hpo --hpo_method optuna --hpo_trials 30 --hpo_startup_trials 5 \
--hpo_space '{"lr":[1e-6,5e-6,1e-5,5e-5],"model_args.width":[256,512],"model_args.dropout":[0.1,0.25,0.4]}'
```

The integration uses `TPESampler`, `study.ask()`, `suggest_categorical()`, and
`study.tell()`. Each trial trains from scratch on the shared data split and
reports its best validation AUC (maximize) or loss (minimize) to Optuna.
OOM and nonfinite trials are marked `FAIL`. Optuna's TPE startup count uses
successful completed trials here; the startup budget is included in the total.
Optuna may suggest repeated combinations: `--hpo_trials` counts training
attempts and is not capped by the number of combinations. Candidate lists must
contain scalar JSON values (numbers, strings, booleans, or null); numeric lists
are categorical, not continuous or log-scale distributions.

The integration keeps the existing JSON report and checkpoint outputs. The
existing per-trial early stopping remains active. The other search methods do
not require Optuna. See the [official ask-and-tell documentation](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html).

### Optional pruning, persistence, and visualization

These switches apply only to `--hpo --hpo_method optuna`. All are disabled by
default and can be enabled independently:

| Option | Default | Effect |
| --- | --- | --- |
| `--hpo_pruner none\|median` | `none` | Enable or disable Optuna median pruning |
| `--hpo_prune_warmup_epochs N` | `5` | Protect the first N epochs from pruning |
| `--hpo_storage [PATH]` | off | Save an Optuna journal text file; no path uses `<ckpt>.hpo.study.log` |
| `--hpo_study_name NAME` | `deeplpa` | Name of the study in the journal file |
| `--hpo_resume` | off | Continue an existing named study |
| `--hpo_visualize` | off | Export interactive HTML figures after search |
| `--hpo_plot_dir PATH` | `<ckpt>.hpo_plots` | Output directory; requires `--hpo_visualize` |

Install visualization dependencies in the training environment:

```bash
python -m pip install 'optuna>=4' plotly scikit-learn
```

Example options to append to your training command:

```bash
--hpo --hpo_method optuna --hpo_trials 30 --hpo_startup_trials 5 \
--hpo_pruner median --hpo_prune_warmup_epochs 5 \
--ckpt outputs/best.pt --hpo_storage --hpo_study_name deeplpa_auc \
--hpo_visualize
```

**Pruning:** Every finite epoch metric is reported via `trial.report()` even when
pruning is disabled. With median pruning enabled, `trial.should_prune()` can
stop a trial after the warmup, once at least `--hpo_startup_trials` successful
trials provide reference results. Epoch steps are zero-based; warmup 5 protects
epochs 1–5. Pruned trials are recorded as `PRUNED` in Optuna and `pruned` in JSON,
not failed. The final epoch is not pruned since its training budget is complete.
Early stopping independently stops a trial when its own metric stops improving.
The checkpoint remains the best finite epoch observed across all trials,
including one later pruned or failed; Optuna's `study.best_trial` considers only
completed trials, so these can differ. See [Optuna pruning](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html).

**Persistence/resume:** To add 20 further trials, repeat the same training/data
settings, journal file path, study name, and checkpoint path, adding:

```bash
--hpo_resume --hpo_trials 20
```

Search uses Optuna's official `JournalStorage` with `JournalFileBackend`.
Operations (parameters, epoch metrics, trial states, and checkpoint metadata)
are appended to a text file as they happen; recovery replays that file.
There is no SQLite database. Windows uses `JournalFileOpenLock` to avoid
requiring symlink privileges. Optuna 4 or newer is required.
See [Optuna journal file storage](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.JournalStorage.html).

With `--ckpt outputs/best.pt --hpo_storage --hpo_visualize`, files are grouped
beside the checkpoint:

```text
outputs/
  best.pt
  best.pt.hpo.json
  best.pt.hpo.study.log
  best.pt.hpo_plots/
    optimization_history.html
    intermediate_values.html
    parallel_coordinate.html
    param_importances.html
```

You may supply a custom file path, such as `--hpo_storage outputs/search.log`.
Omitting `--hpo_storage` keeps the study in memory. Existing SQLite files cannot
be resumed through this backend; choose a new journal file for a new search.

Resume requires the existing journal file and its saved best checkpoint.
It restores historical JSON trial records and the global checkpoint score, so
inferior new trials do not overwrite the old winner. Data/model/search settings
are checked for consistency; keep the underlying dataset and checkpoint file
unchanged. This is sequential, exclusive study use: do not run another process
against the same study while resuming. Interrupted `RUNNING` trials are marked
`FAIL` and replaced by fresh trials; model/optimizer epoch state is not resumed.
The sampler RNG is initialized again on resume, so the suggestion sequence is
not guaranteed to match an uninterrupted run. Reusing an existing study name
without `--hpo_resume` fails rather than silently merging studies.

**Visualization:** Open the exported files in a browser:

- `optimization_history.html`: trial scores and optimization progress.
- `intermediate_values.html`: epoch metrics, including pruned trials.
- `parallel_coordinate.html`: relationships between parameters and scores.
- `param_importances.html`: estimated parameter importance.

Plotly is embedded in each HTML file for offline viewing. Figures include the
entire persisted study on resume. Sparse or constant results may prevent some
plots (especially importance estimation); skipped plots and reasons are recorded
in the JSON `plots` field. Visualization is performed after search, not live.
See [Optuna visualization](https://optuna.readthedocs.io/en/stable/reference/visualization/index.html).

### Built-in categorical TPE

For adaptive search, add `--hpo_method tpe --hpo_startup_trials 5` to the example
below. No additional package is required. This is a simplified categorical TPE
implementation for the existing finite candidate lists, not an Optuna integration.
After five trials with finite scores complete successfully, it ranks their best
validation scores, assigns the top 20% to the good group (rounded up), and fits
independent categorical frequencies with uniform pseudocounts for each parameter.
It samples up to 24 unused combinations and chooses the highest summed log
good/bad density ratio. Failed or nonfinite trials do not inform the densities.
Until enough successful trials exist, it samples randomly. Startup must be at
least two; a budget no greater than startup provides no adaptive phase.

The built-in `random`, `grid`, and `tpe` methods use at most `--hpo_trials` distinct combinations. Grid search
is exhaustive only when the budget covers the full Cartesian product; a smaller
budget evaluates its first combinations. TPE treats even numeric candidates as
categories: it does not interpolate values, model parameter interactions, or
guarantee better results than random search. Its density-ratio principle follows
[TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html).

```bash
python main.py \
  --data_dir /path/to/data --tree_name Tree1 \
  --scalar_branches mass,chisq --target_branches is_sig,is_bkg \
  --num_classes 2 --model_class model.DRNmodfied.DRNmod \
  --model_args '{"width":256,"depth":4,"dropout":0.25}' \
  --loss_class model.FocalLoss.FocalLoss \
  --loss_args '{"signal_class":0,"alpha":0.9,"gamma":2.0}' \
  --hpo --hpo_trials 12 --hpo_metric auc \
  --hpo_space '{"lr":[1e-6,5e-6,1e-5],"model_args.width":[256,512],"loss_args.alpha":[0.85,0.9]}' \
  --epochs 100 --early_stop_patience 5 --val_ratio 0.25 \
  --ckpt best_hpo.pt
```

Search keys can be `lr`, `weight_decay`, `batch_size`, `model_args.<argument>`,
or `loss_args.<argument>`. Each value is a non-empty JSON list. Parameters absent
from the search space keep their command-line values. Without `--hpo_space`,
search tries learning rates at 0.3, 1, and 3 times `--lr`, and weight decay values
of zero and `--weight_decay`.

`--hpo_metric auto` maximizes validation AUC for classification and minimizes
validation loss for regression. Binary AUC treats `--pos_class` as positive;
multiclass AUC uses macro-averaged one-vs-rest. Early stopping during search uses
this same metric, with the existing patience and minimum-delta options. Each
trial has a budget of `--epochs`. `--hpo_seed` controls candidate sampling and
model initialization; `--split_seed` controls the shared data split.

Results are saved after every trial to `<ckpt>.hpo.json`, or `--hpo_results`.
The JSON includes the method, search space, winning model/loss arguments, optimizer settings, score,
epoch, and the results of all trials. For inference, use the winning
`model_class`, `model_args`, `loss_class`, and `loss_args` from `best.parameters`
alongside `--mode infer --ckpt best_hpo.pt` and the original data settings.

Search finishes after saving the winning checkpoint; it does not run the normal
post-training plots or prediction exports. `--load_ckpt` is not supported in
search mode. GPU out-of-memory trials are recorded and skipped; invalid model
or loss arguments fail explicitly. Use a separate test sample to assess the
chosen configuration. Search finds the best evaluated configuration,
not a guaranteed global optimum.
