import json
import math
import os
import random


def validate_space(space, trials):
    if not isinstance(space, dict) or not space:
        raise ValueError("Search space must be a non-empty JSON object")
    if trials <= 0:
        raise ValueError("--hpo_trials must be positive")
    keys, values = [], []
    for key, candidates in space.items():
        if key not in ("lr", "weight_decay", "batch_size") and not (
            isinstance(key, str) and key.startswith(("model_args.", "loss_args."))
            and len(key.split(".")) == 2 and key.split(".")[1]
        ):
            raise ValueError(f"Unsupported search parameter: {key}")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Candidates for {key} must be a non-empty list")
        unique = []
        for value in candidates:
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid JSON candidate for {key}") from error
            if key in ("lr", "weight_decay", "batch_size"):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{key} candidates must be finite numbers")
                if key == "batch_size" and (not isinstance(value, int) or value <= 0):
                    raise ValueError("batch_size candidates must be positive integers")
                if (key == "lr" and value <= 0) or (key == "weight_decay" and value < 0):
                    raise ValueError(f"Invalid candidate for {key}: {value}")
            if value not in unique:
                unique.append(value)
        keys.append(key)
        values.append(unique)
    return keys, values


def decode_configuration(index, keys, values):
    selected = []
    for candidates in reversed(values):
        index, offset = divmod(index, len(candidates))
        selected.append(candidates[offset])
    return dict(zip(keys, reversed(selected)))


def sample_configurations(space, trials, seed):
    """Sample a finite Cartesian search space without repeating configurations."""
    keys, values = validate_space(space, trials)
    total = math.prod(len(v) for v in values)
    indices = random.Random(seed).sample(range(total), min(trials, total))
    return [decode_configuration(index, keys, values) for index in indices]


def iter_configurations(space, trials, seed, method, startup, history, metric, context=None):
    """Yield configurations; TPE reads scores appended to history after each yield."""
    keys, values = validate_space(space, trials)
    total = math.prod(len(v) for v in values)
    if method not in ("random", "grid", "tpe", "optuna"):
        raise ValueError(f"Unknown search method: {method}")
    if startup < 2:
        raise ValueError("--hpo_startup_trials must be at least 2")
    if method == "optuna":
        try:
            import optuna
        except ImportError as error:
            raise ImportError("Optuna search requires the optional dependency: python -m pip install optuna") from error
        for key, candidates in zip(keys, values):
            if any(value is not None and not isinstance(value, (bool, int, float, str))
                   for value in candidates):
                raise ValueError(f"Optuna candidates for {key} must be scalar JSON values")
        study = context["study"] if context else optuna.create_study(
            direction="maximize" if metric == "auc" else "minimize",
            sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=startup))
        for _ in range(trials):
            trial = study.ask()
            config = {key: trial.suggest_categorical(key, candidates)
                      for key, candidates in zip(keys, values)}
            if context is not None:
                context["trial"] = trial
            yield config
            record = history[-1]
            trial.set_user_attr("record", record)
            if record["status"] == "pruned":
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            elif record["status"] == "completed" and record["score"] is not None and math.isfinite(record["score"]):
                study.tell(trial, record["score"])
            else:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
        return
    if method == "random":
        yield from sample_configurations(space, trials, seed)
        return
    if method == "grid":
        for index in range(min(trials, total)):
            yield decode_configuration(index, keys, values)
        return
    rng = random.Random(seed)
    used, selected = set(), []
    for _ in range(min(trials, total)):
        # Sample ranks among unused combinations without enumerating the space.
        ranked_used = sorted(used)
        pool = []
        for rank in rng.sample(range(total - len(used)), min(24, total - len(used))):
            index = rank
            for excluded in ranked_used:
                if excluded > index:
                    break
                index += 1
            pool.append((index, decode_configuration(index, keys, values)))
        observations = [(config, record["score"]) for config, record in zip(selected, history)
                        if record["status"] == "completed" and record["score"] is not None
                        and math.isfinite(record["score"])]
        if len(observations) < startup:
            index, config = pool[0]
        else:
            observations.sort(key=lambda item: item[1], reverse=metric == "auc")
            cutoff = max(1, math.ceil(len(observations) * 0.2))
            good, bad = observations[:cutoff], observations[cutoff:]

            def density_ratio(candidate):
                score = 0.0
                for key, candidates in zip(keys, values):
                    # Categorical densities with a uniform pseudocount per value.
                    good_count = 1 + sum(c[key] == candidate[key] for c, _ in good)
                    bad_count = 1 + sum(c[key] == candidate[key] for c, _ in bad)
                    score += math.log(good_count / (len(good) + len(candidates)))
                    score -= math.log(bad_count / (len(bad) + len(candidates)))
                return score

            index, config = max(pool, key=lambda item: density_ratio(item[1]))
        used.add(index)
        selected.append(config)
        yield config


def prepare_optuna(args, metric, signature):
    import optuna

    warmup = getattr(args, "hpo_prune_warmup_epochs", 5)
    if warmup < 0:
        raise ValueError("--hpo_prune_warmup_epochs must be nonnegative")
    pruner = (optuna.pruners.MedianPruner(n_startup_trials=args.hpo_startup_trials,
                                        n_warmup_steps=warmup)
              if getattr(args, "hpo_pruner", "none") == "median" else optuna.pruners.NopPruner())
    study_file = getattr(args, "hpo_storage", "")
    resume = getattr(args, "hpo_resume", False)
    if resume and not study_file:
        raise ValueError("--hpo_resume requires --hpo_storage")
    storage = None
    if study_file:
        from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

        study_file = os.path.abspath(args.ckpt + ".hpo.study.log" if study_file == "auto" else study_file)
        if os.path.normcase(study_file) in (
            os.path.normcase(os.path.abspath(args.ckpt)),
            os.path.normcase(os.path.abspath(args.hpo_results or args.ckpt + ".hpo.json")),
        ):
            raise ValueError("Study file must differ from checkpoint and results paths")
        if resume and not os.path.isfile(study_file):
            raise FileNotFoundError(f"Study file does not exist: {study_file}")
        if os.path.isfile(study_file):
            with open(study_file, "rb") as stream:
                if stream.read(16) == b"SQLite format 3\x00":
                    raise ValueError("Existing SQLite databases cannot be used as journal files; choose a new file")
        os.makedirs(os.path.dirname(study_file), exist_ok=True)
        storage = optuna.storages.JournalStorage(
            JournalFileBackend(study_file, lock_obj=JournalFileOpenLock(study_file)))
    sampler = optuna.samplers.TPESampler(seed=args.hpo_seed, n_startup_trials=args.hpo_startup_trials)
    direction = "maximize" if metric == "auc" else "minimize"
    if resume:
        study = optuna.load_study(study_name=args.hpo_study_name, storage=storage,
                                  sampler=sampler, pruner=pruner)
        if study.direction.name.lower() != direction or study.user_attrs.get("signature") != signature:
            raise ValueError("Study settings differ; use the original search/data settings or a new study name")
        best = study.user_attrs.get("checkpoint_best")
        if best is not None and not os.path.isfile(args.ckpt):
            raise ValueError("Resume requires the existing best checkpoint at --ckpt")
        # Resume is for exclusive sequential use; interrupted trials restart as new trials.
        for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    else:
        study = optuna.create_study(study_name=args.hpo_study_name if storage else None,
                                   storage=storage, direction=direction, sampler=sampler, pruner=pruner)
        study.set_user_attr("signature", signature)
    return {"study": study}, study_file


def export_optuna_plots(study, directory):
    import optuna

    os.makedirs(directory, exist_ok=True)
    plots = {
        "optimization_history": optuna.visualization.plot_optimization_history,
        "intermediate_values": optuna.visualization.plot_intermediate_values,
        "parallel_coordinate": optuna.visualization.plot_parallel_coordinate,
        "param_importances": optuna.visualization.plot_param_importances,
    }
    outputs = {}
    for name, plot in plots.items():
        try:
            path = os.path.abspath(os.path.join(directory, name + ".html"))
            plot(study).write_html(path, include_plotlyjs=True)
            outputs[name] = {"status": "saved", "path": path}
        except (ValueError, RuntimeError, ImportError) as error:
            # Sparse/constant trial results may not permit importance estimation.
            outputs[name] = {"status": "skipped", "reason": str(error)}
            print(f"[HPO] Plot {name} skipped: {error}")
    return outputs


def run_hyperparameter_search(args, in_dim, train_loader, val_loader, device,
                              import_class, train_one_epoch, evaluate):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    if train_loader is None or val_loader is None or not len(train_loader.dataset) or not len(val_loader.dataset):
        raise ValueError("Hyperparameter search requires non-empty training and validation sets")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive for hyperparameter search")
    if args.load_ckpt:
        raise ValueError("Hyperparameter search starts fresh models; omit --load_ckpt")
    metric = args.hpo_metric
    if metric == "auto":
        metric = "auc" if args.task == "classification" else "loss"
    if metric == "auc" and args.task != "classification":
        raise ValueError("AUC search is only available for classification")
    space = json.loads(args.hpo_space) if args.hpo_space else {
        "lr": [args.lr * 0.3, args.lr, args.lr * 3],
        "weight_decay": [0.0, args.weight_decay],
    }
    method = getattr(args, "hpo_method", "random")
    startup = getattr(args, "hpo_startup_trials", 5)
    if startup < 2:
        raise ValueError("--hpo_startup_trials must be at least 2")
    optuna_options = (getattr(args, "hpo_pruner", "none") != "none" or
                      getattr(args, "hpo_storage", "") or getattr(args, "hpo_resume", False) or
                      getattr(args, "hpo_visualize", False) or getattr(args, "hpo_plot_dir", ""))
    if method != "optuna" and optuna_options:
        raise ValueError("Pruning, study storage/resume and visualization require --hpo_method optuna")
    if getattr(args, "hpo_plot_dir", "") and not getattr(args, "hpo_visualize", False):
        raise ValueError("--hpo_plot_dir requires --hpo_visualize")
    if getattr(args, "hpo_visualize", False):
        try:
            import plotly
        except ImportError as error:
            raise ImportError("Optuna visualization requires: python -m pip install plotly scikit-learn") from error
    _, values = validate_space(space, args.hpo_trials)
    trial_count = args.hpo_trials if method == "optuna" else min(args.hpo_trials, math.prod(len(v) for v in values))
    base_model_args = json.loads(args.model_args)
    base_loss_args = json.loads(args.loss_args)
    results_path = args.hpo_results or args.ckpt + ".hpo.json"
    if os.path.abspath(results_path) == os.path.abspath(args.ckpt):
        raise ValueError("Checkpoint and search results paths must differ")
    for path in (args.ckpt, results_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    report = {"method": method, "startup_trials": startup, "search_space": space,
              "metric": metric, "direction": "maximize" if metric == "auc" else "minimize",
              "seed": args.hpo_seed, "split_seed": args.split_seed,
              "checkpoint": args.ckpt, "best": None, "trials": []}
    best_score = -math.inf if metric == "auc" else math.inf
    context = None
    if method == "optuna":
        signature = {"space": space, "metric": metric, "seed": args.hpo_seed,
                     "split_seed": args.split_seed, "checkpoint": os.path.abspath(args.ckpt),
                     "model_class": args.model_class, "model_args": base_model_args,
                     "loss_class": args.loss_class, "loss_args": base_loss_args,
                     "in_dim": in_dim, "train_size": len(train_loader.dataset),
                     "val_size": len(val_loader.dataset)}
        signature["training"] = {key: getattr(args, key, None) for key in (
            "task", "num_classes", "lr", "weight_decay", "batch_size", "epochs",
            "early_stop_patience", "early_stop_min_delta", "pos_class", "threshold",
            "data_dir", "tree_name", "scalar_branches", "target_branches", "val_ratio",
            "array_spec", "pad_value", "reg_target_names", "normalize", "scaler_file")}
        try:
            context, study_file = prepare_optuna(args, metric, signature)
        except ImportError as error:
            raise ImportError("Optuna search requires: python -m pip install optuna") from error
        study = context["study"]
        report.update(study_name=study.study_name, storage=study_file,
                      pruner=getattr(args, "hpo_pruner", "none"),
                      prune_warmup_epochs=getattr(args, "hpo_prune_warmup_epochs", 5))
        report["best"] = study.user_attrs.get("checkpoint_best")
        if report["best"] is not None:
            best_score = report["best"]["score"]
        for trial in study.trials:
            record = trial.user_attrs.get("record")
            if record is None:
                record = {"trial": trial.number + 1, "optuna_number": trial.number,
                          "parameters": trial.params, "score": None, "best_epoch": None,
                          "epochs_run": 0, "status": "interrupted"}
            elif trial.state.name == "FAIL" and record["status"] == "completed":
                record["status"] = "interrupted"
            report["trials"].append(record)
    model_class = import_class(args.model_class)
    loss_class = import_class(args.loss_class)

    configurations = iter_configurations(space, args.hpo_trials, args.hpo_seed,
                                         method, startup, report["trials"], metric, context)
    for trial_id, config in enumerate(configurations, 1):
        model_args, loss_args = dict(base_model_args), dict(base_loss_args)
        training = {"lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size}
        for key, value in config.items():
            if key.startswith("model_args."):
                model_args[key.split(".", 1)[1]] = value
            elif key.startswith("loss_args."):
                loss_args[key.split(".", 1)[1]] = value
            else:
                training[key] = value
        parameters = {**training, "model_class": args.model_class, "model_args": model_args,
                      "loss_class": args.loss_class, "loss_args": loss_args}
        record = {"trial": trial_id, "parameters": parameters, "score": None,
                  "best_epoch": None, "epochs_run": 0, "status": "no_finite_metric"}
        optuna_trial = context["trial"] if context else None
        if optuna_trial is not None:
            record["trial"] = optuna_trial.number + 1
            record["optuna_number"] = optuna_trial.number
        print(f"[HPO] {method} trial {trial_id}/{trial_count}: {config}")
        random.seed(args.hpo_seed)
        np.random.seed(args.hpo_seed)
        torch.manual_seed(args.hpo_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.hpo_seed)
        model = criterion = optimizer = trial_train = trial_val = None
        try:
            model = model_class(in_dim=in_dim, num_classes=args.num_classes,
                                task=args.task, **model_args).to(device)
            criterion = loss_class(**loss_args).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=training["lr"],
                                          weight_decay=training["weight_decay"])
            trial_train = DataLoader(train_loader.dataset, batch_size=training["batch_size"],
                                     shuffle=True, num_workers=args.num_workers,
                                     pin_memory=device.type == "cuda")
            trial_val = DataLoader(val_loader.dataset, batch_size=training["batch_size"],
                                   shuffle=False, num_workers=args.num_workers,
                                   pin_memory=device.type == "cuda")
            # Early stopping compares with this trial's own best, not the global best.
            patience_score = -math.inf if metric == "auc" else math.inf
            stale_epochs = 0
            for epoch in range(1, args.epochs + 1):
                train_one_epoch(model, trial_train, optimizer, criterion, device, args.task)
                metrics = evaluate(model, trial_val, criterion, device, args.task,
                                   args.pos_class, args.threshold)
                score = metrics.get(metric)
                record["epochs_run"] = epoch
                finite = score is not None and math.isfinite(float(score))
                if finite:
                    score = float(score)
                    better = record["score"] is None or (score > record["score"] if metric == "auc" else score < record["score"])
                    if better:
                        record.update(score=score, best_epoch=epoch, status="completed")
                    global_better = score > best_score if metric == "auc" else score < best_score
                    if global_better:
                        torch.save(model.state_dict(), args.ckpt)
                        best_score = score
                        report["best"] = {"trial": trial_id, "epoch": epoch,
                                          "score": score, "parameters": parameters}
                        if context:
                            report["best"]["trial"] = record["trial"]
                            study.set_user_attr("checkpoint_best", report["best"])
                    if optuna_trial is not None:
                        optuna_trial.report(score, step=epoch - 1)
                improved = finite and (score > patience_score + args.early_stop_min_delta if metric == "auc"
                                       else score < patience_score - args.early_stop_min_delta)
                if improved:
                    patience_score, stale_epochs = score, 0
                else:
                    stale_epochs += 1
                print(f"[HPO] Trial {trial_id}, epoch {epoch}: val_{metric}={score}")
                if optuna_trial is not None:
                    optuna_trial.set_user_attr("record", record)
                    if finite and epoch < args.epochs and optuna_trial.should_prune():
                        record["status"] = "pruned"
                        print(f"[HPO] Trial {trial_id} pruned at epoch {epoch}")
                        break
                if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
                    break
        except torch.cuda.OutOfMemoryError:
            record["status"] = "out_of_memory"
            print(f"[HPO] Trial {trial_id} stopped: GPU out of memory")
        finally:
            del model, criterion, optimizer, trial_train, trial_val
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report["trials"].append(record)
        with open(results_path, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    if context and getattr(args, "hpo_visualize", False):
        directory = getattr(args, "hpo_plot_dir", "") or args.ckpt + ".hpo_plots"
        report["plots"] = export_optuna_plots(study, directory)
        with open(results_path, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    if report["best"] is None:
        raise RuntimeError(f"No trial produced a finite validation {metric}; see {results_path}")
    print(f"[HPO] Best validation {metric}: {report['best']['score']}; checkpoint: {args.ckpt}")
    print(f"[HPO] Parameters and trial results: {results_path}")
    return report
