import json
import ast
import argparse
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
try:
    import optuna
except ImportError:
    optuna = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.hyperparameter_search import sample_configurations, iter_configurations, run_hyperparameter_search, export_optuna_plots


class SearchSpaceTests(unittest.TestCase):
    def test_storage_switch_accepts_automatic_and_explicit_paths(self):
        source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
        function = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == "parse_args")
        namespace = {"argparse": argparse}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "main.py", "exec"), namespace)
        required = ["main.py", "--data_dir", "/data", "--num_classes", "2",
                    "--model_class", "Model", "--loss_class", "Loss"]
        for flags, expected in (([], ""), (["--hpo_storage", "--hpo_resume"], "auto"),
                                (["--hpo_storage", "outputs/search.log"], "outputs/search.log")):
            with self.subTest(flags=flags), patch.object(sys, "argv", required + flags):
                self.assertEqual(namespace["parse_args"]().hpo_storage, expected)

    def test_grid_visits_all_combinations_once_in_order(self):
        configs = list(iter_configurations({"lr": [0.1, 0.1, 0.2], "batch_size": [4, 8]},
                                          20, 42, "grid", 5, [], "auc"))
        self.assertEqual(configs, [{"lr": lr, "batch_size": batch}
                                  for lr in (0.1, 0.2) for batch in (4, 8)])

    def test_tpe_responds_to_scores_and_metric_direction(self):
        space = {"lr": [0.1, 0.2], "model_args.width": list(range(10))}

        def run(seed, metric, reverse=False):
            history, configs = [], []
            for config in iter_configurations(space, 3, seed, "tpe", 2, history, metric):
                configs.append(config)
                score = config["lr"] * (-1 if reverse else 1)
                history.append({"score": score, "status": "completed"})
            return configs

        seed = next(s for s in range(100) if run(s, "auc")[0]["lr"] != run(s, "auc")[1]["lr"])
        maximize, minimize = run(seed, "auc"), run(seed, "loss")
        self.assertEqual(maximize[:2], minimize[:2])
        self.assertEqual(maximize[2]["lr"], 0.2)
        self.assertEqual(minimize[2]["lr"], 0.1)
        self.assertEqual(maximize, run(seed, "auc"))
        self.assertEqual(minimize, run(seed, "auc", reverse=True))

    def test_tpe_exhausts_space_without_duplicates_or_valid_scores(self):
        history = []
        configs = []
        for config in iter_configurations({"lr": [0.1, 0.2, 0.3]}, 20, 42,
                                          "tpe", 2, history, "auc"):
            configs.append(config)
            history.append({"score": None, "status": "out_of_memory"})
        self.assertEqual(len(configs), 3)
        self.assertEqual(len({c["lr"] for c in configs}), 3)

    def test_sampling_is_reproducible_unique_and_bounded(self):
        space = {"lr": [0.1, 0.1, 0.2], "model_args.width": [16, 32]}
        configs = sample_configurations(space, 100, 42)
        self.assertEqual(configs, sample_configurations(space, 100, 42))
        self.assertEqual(len(configs), 4)
        self.assertEqual(len({json.dumps(c, sort_keys=True) for c in configs}), 4)

    def test_invalid_parameters_fail_before_training(self):
        for space in ({"split_seed": [1]}, {"lr": [0]}, {"batch_size": [1.5]},
                      {"lr": []}, {"weight_decay": [-1]}, {"lr": [math.nan]}):
            with self.subTest(space=space), self.assertRaises(ValueError):
                sample_configurations(space, 3, 42)

    def test_optuna_dependency_error_explains_installation(self):
        with patch.dict(sys.modules, {"optuna": None}), self.assertRaisesRegex(ImportError, "pip install optuna"):
            next(iter_configurations({"lr": [0.1]}, 3, 42, "optuna", 2, [], "auc"))


@unittest.skipIf(optuna is None, "Optional Optuna dependency is not installed")
class OptunaSearchTests(unittest.TestCase):
    def test_pruned_trials_have_pruned_state(self):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=42))
        history = []
        context = {"study": study}
        for config in iter_configurations({"lr": [0.1]}, 1, 42, "optuna", 2, history, "auc", context):
            context["trial"].report(0.2, 0)
            history.append({"status": "pruned", "score": 0.2})
        self.assertEqual(study.trials[0].state, optuna.trial.TrialState.PRUNED)
        self.assertEqual(study.trials[0].intermediate_values, {0: 0.2})

    def test_html_visualization_exports_real_optuna_figures(self):
        try:
            import plotly
            import sklearn
        except ImportError:
            self.skipTest("Optional visualization dependencies are not installed")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=42))
        for index in range(6):
            trial = study.ask()
            trial.suggest_categorical("lr", [0.1, 0.2, 0.3])
            trial.report(index / 10, 0)
            study.tell(trial, index / 10)
        with tempfile.TemporaryDirectory() as directory:
            outputs = export_optuna_plots(study, directory)
            self.assertEqual(len(outputs), 4)
            for output in outputs.values():
                self.assertEqual(output["status"], "saved")
                html = Path(output["path"]).read_text(encoding="utf-8")
                self.assertIn("Plotly.newPlot", html)
                self.assertNotIn('<script src="https://cdn.plot.ly', html)
    def test_real_sampler_receives_scores_direction_and_seed(self):
        for metric, direction in (("auc", "maximize"), ("loss", "minimize")):
            def run():
                study = optuna.create_study(direction=direction,
                    sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=2))
                history, configs = [], []
                with patch.object(optuna, "create_study", return_value=study):
                    for config in iter_configurations({"lr": [0.1, 0.2, 0.3]}, 12, 42,
                                                      "optuna", 2, history, metric):
                        configs.append(config)
                        history.append({"status": "completed", "score": config["lr"]})
                self.assertEqual(len(study.trials), 12)
                self.assertEqual([t.value for t in study.trials], [c["lr"] for c in configs])
                expected = (max if metric == "auc" else min)(c["lr"] for c in configs)
                self.assertEqual(study.best_value, expected)
                return configs
            with self.subTest(metric=metric):
                self.assertEqual(run(), run())

    def test_failed_trials_are_not_reported_as_successful_scores(self):
        study = optuna.create_study()
        history = []
        with patch.object(optuna, "create_study", return_value=study):
            for config in iter_configurations({"lr": [0.1]}, 3, 42, "optuna", 2, history, "loss"):
                # Even an OOM after a finite epoch must be marked failed.
                history.append({"status": "out_of_memory", "score": 0.5})
        self.assertEqual(len(study.trials), 3)
        self.assertTrue(all(t.state == optuna.trial.TrialState.FAIL and t.value is None
                            for t in study.trials))

    def test_nested_candidates_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "scalar JSON"):
            next(iter_configurations({"model_args.width": [[16, 32]]}, 3, 42,
                                     "optuna", 2, [], "auc"))


class SearchTrainingTests(unittest.TestCase):
    def run_search(self, directory, task="classification", metric="auto", nan=False, oom=False, method="random", options=None, scores=None):
        train_dataset, val_dataset = object(), object()
        train_dataset, val_dataset = [train_dataset] * 8, [val_dataset] * 4

        model_count = 0

        class Model:
            def __init__(self, **kwargs):
                nonlocal model_count
                self.index = model_count
                model_count += 1
                self.epoch = 0
            def to(self, device):
                return self
            def parameters(self):
                return []
            def state_dict(self):
                return {"lr": self.lr, "epoch": self.epoch}

        class Loss:
            def to(self, device):
                return self

        class Loader:
            def __init__(self, dataset, **kwargs):
                self.dataset = dataset

        def train(model, loader, optimizer, criterion, device, task):
            self.assertIs(loader.dataset, train_dataset)
            if oom and optimizer.lr == 0.3:
                raise MemoryError("simulated GPU exhaustion")
            model.lr = optimizer.lr
            model.epoch += 1

        def evaluate(model, loader, criterion, device, task, pos_class, threshold):
            self.assertIs(loader.dataset, val_dataset)
            score = {0.1: 0.7, 0.2: 0.9, 0.3: 0.8}[model.lr]
            # Later epochs are worse: checkpoint must preserve the best epoch.
            score -= 0.1 * (model.epoch - 1)
            if scores is not None:
                score = scores[model.index]
            return {"auc": math.nan if nan else score,
                    "loss": math.nan if nan else 1 - score}

        def save(state, path):
            Path(path).write_text(json.dumps(state), encoding="utf-8")

        torch = SimpleNamespace(
            manual_seed=lambda seed: None,
            cuda=SimpleNamespace(is_available=lambda: False, OutOfMemoryError=MemoryError),
            optim=SimpleNamespace(AdamW=lambda params, **kwargs: SimpleNamespace(**kwargs)),
            save=save,
        )
        args = SimpleNamespace(hpo_metric=metric, hpo_space='{"lr":[0.1,0.2,0.3]}',
                               hpo_method=method, hpo_startup_trials=2,
                               hpo_trials=3, hpo_seed=42, split_seed=42, epochs=2,
                               load_ckpt="", model_args="{}", loss_args="{}",
                               hpo_results="", ckpt=str(Path(directory) / "best.pt"),
                               model_class="Model", loss_class="Loss", lr=0.1,
                               weight_decay=0, batch_size=4, num_classes=2, task=task,
                               num_workers=0, pos_class=0, threshold=0.5,
                               early_stop_patience=1, early_stop_min_delta=0)
        if options:
            vars(args).update(options)
        with patch.dict(sys.modules, {"torch": torch, "torch.utils.data": SimpleNamespace(DataLoader=Loader)}):
            report = run_hyperparameter_search(
                args, 3, SimpleNamespace(dataset=train_dataset), SimpleNamespace(dataset=val_dataset),
                SimpleNamespace(type="cpu"), lambda name: Model if name == "Model" else Loss,
                train, evaluate)
        return report, args

    def test_classification_saves_best_trial_and_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            report, args = self.run_search(directory)
            self.assertEqual(report["metric"], "auc")
            self.assertEqual(report["best"]["parameters"]["lr"], 0.2)
            self.assertEqual(json.loads(Path(args.ckpt).read_text()), {"lr": 0.2, "epoch": 1})
            self.assertEqual(len(json.loads(Path(args.ckpt + ".hpo.json").read_text())["trials"]), 3)
            self.assertTrue(all(t["epochs_run"] == 2 for t in report["trials"]))

    def test_new_methods_integrate_with_training_and_checkpoint(self):
        for method in ("grid", "tpe"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                report, args = self.run_search(directory, method=method)
                self.assertEqual(report["method"], method)
                self.assertEqual(report["best"]["parameters"]["lr"], 0.2)
                self.assertEqual(len(report["trials"]), 3)
                self.assertEqual(json.loads(Path(args.ckpt).read_text()), {"lr": 0.2, "epoch": 1})

    @unittest.skipIf(optuna is None, "Optional Optuna dependency is not installed")
    def test_optuna_integrates_with_training_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            report, args = self.run_search(directory, method="optuna")
            self.assertEqual(report["method"], "optuna")
            self.assertEqual(report["best"]["score"], max(t["score"] for t in report["trials"]))
            self.assertEqual(len(report["trials"]), 3)
            checkpoint = json.loads(Path(args.ckpt).read_text())
            self.assertEqual(checkpoint["lr"], report["best"]["parameters"]["lr"])
            self.assertEqual(checkpoint["epoch"], report["best"]["epoch"])

    @unittest.skipIf(optuna is None, "Optional Optuna dependency is not installed")
    def test_median_pruning_respects_warmup_and_can_be_disabled(self):
        for task in ("classification", "regression"):
            for pruner in ("none", "median"):
                with self.subTest(task=task, pruner=pruner), tempfile.TemporaryDirectory() as directory:
                    report, args = self.run_search(directory, task=task, method="optuna", scores=[0.9, 0.8, 0.1],
                        options={"hpo_pruner": pruner, "hpo_prune_warmup_epochs": 1,
                                 "epochs": 4, "early_stop_patience": 0})
                    self.assertEqual(report["trials"][2]["status"], "pruned" if pruner == "median" else "completed")
                    self.assertEqual(report["trials"][2]["epochs_run"], 2 if pruner == "median" else 4)
                    self.assertAlmostEqual(report["best"]["score"], 0.9 if task == "classification" else 0.1)
                    self.assertFalse(Path(args.ckpt + ".hpo_plots").exists())

    @unittest.skipIf(optuna is None, "Optional Optuna dependency is not installed")
    def test_visualization_switch_exports_and_records_outputs(self):
        try:
            import plotly
        except ImportError:
            self.skipTest("Optional Plotly dependency is not installed")
        with tempfile.TemporaryDirectory() as directory:
            outputs = {"optimization_history": {"status": "saved", "path": "history.html"}}
            with patch("utils.hyperparameter_search.export_optuna_plots", return_value=outputs) as export:
                report, args = self.run_search(directory, method="optuna", options={"hpo_visualize": True})
            export.assert_called_once()
            self.assertEqual(export.call_args.args[1], args.ckpt + ".hpo_plots")
            self.assertEqual(report["plots"], outputs)
            self.assertEqual(json.loads(Path(args.ckpt + ".hpo.json").read_text())["plots"], outputs)

    @unittest.skipIf(optuna is None, "Optional Optuna dependency is not installed")
    def test_file_resume_preserves_best_checkpoint_and_trial_history(self):
        with tempfile.TemporaryDirectory() as directory:
            options = {"hpo_storage": "auto",
                       "hpo_study_name": "resume_test", "hpo_trials": 2}
            initial, args = self.run_search(directory, method="optuna", options=options, scores=[0.9, 0.8])
            checkpoint = Path(args.ckpt).read_text()
            study_file = Path(args.ckpt + ".hpo.study.log")
            self.assertEqual(initial["storage"], str(study_file.resolve()))
            self.assertTrue(study_file.exists())
            self.assertTrue(all(isinstance(json.loads(line), dict)
                                for line in study_file.read_text(encoding="utf-8").splitlines()))
            resumed, _ = self.run_search(directory, method="optuna",
                options={**options, "hpo_resume": True}, scores=[0.1, 0.2])
            self.assertEqual(len(resumed["trials"]), 4)
            self.assertEqual([t["optuna_number"] for t in resumed["trials"]], [0, 1, 2, 3])
            self.assertEqual(initial["best"], resumed["best"])
            self.assertEqual(Path(args.ckpt).read_text(), checkpoint)
            storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(
                str(study_file), lock_obj=optuna.storages.journal.JournalFileOpenLock(str(study_file))))
            study = optuna.load_study(study_name="resume_test", storage=storage)
            self.assertEqual(len(study.trials), 4)
            self.assertEqual(study.best_value, 0.9)
            # Simulate interruption before completing a trial.
            study.ask()
            self.run_search(directory, method="optuna", options={**options, "hpo_resume": True})
            self.assertEqual(study.trials[4].state, optuna.trial.TrialState.FAIL)
            with self.assertRaisesRegex(ValueError, "settings differ"):
                self.run_search(directory, method="optuna", options={**options, "hpo_resume": True, "split_seed": 99})

    @unittest.skipIf(optuna is None, "Optional Optuna dependency is not installed")
    def test_custom_file_path_and_missing_resume_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "search" / "study.log")
            options = {"hpo_storage": path, "hpo_study_name": "custom_test"}
            with self.assertRaisesRegex(FileNotFoundError, "Study file does not exist"):
                self.run_search(directory, method="optuna", options={**options, "hpo_resume": True})
            self.assertFalse(Path(path).exists())
            report, _ = self.run_search(directory, method="optuna", options=options)
            self.assertEqual(report["storage"], path)
            self.assertTrue(Path(path).exists())

    @unittest.skipIf(optuna is None, "Optional Optuna dependency is not installed")
    def test_existing_sqlite_file_is_not_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.db"
            original = b"SQLite format 3\x00" + b"test database contents"
            path.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "Existing SQLite databases"):
                self.run_search(directory, method="optuna", options={"hpo_storage": str(path)})
            self.assertEqual(path.read_bytes(), original)

    def test_optuna_only_switches_reject_other_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            for options in ({"hpo_pruner": "median"}, {"hpo_visualize": True},
                            {"hpo_storage": str(Path(directory) / "study.log")}, {"hpo_resume": True}):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, "require --hpo_method optuna"):
                    self.run_search(directory, options=options)

    def test_out_of_memory_candidate_does_not_abort_search(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _ = self.run_search(directory, oom=True)
            self.assertEqual(report["trials"][0]["status"], "out_of_memory")
            self.assertEqual(report["best"]["parameters"]["lr"], 0.2)

    def test_regression_minimizes_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _ = self.run_search(directory, task="regression")
            self.assertEqual(report["direction"], "minimize")
            self.assertEqual(report["best"]["parameters"]["lr"], 0.2)

    def test_all_nonfinite_trials_do_not_create_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "No trial produced"):
                self.run_search(directory, nan=True)
            self.assertFalse((Path(directory) / "best.pt").exists())
            report = json.loads((Path(directory) / "best.pt.hpo.json").read_text())
            self.assertIsNone(report["best"])


class BinaryAUCTests(unittest.TestCase):
    def test_evaluation_maps_selected_class_to_positive(self):
        # Exercise the actual AUC expression without importing the full ROOT stack.
        source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
        assignment = next(n for n in ast.walk(function) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "auc" for t in n.targets))
        expression = compile(ast.Expression(assignment.value), "main.py", "eval")
        labels = np.array([0, 1, 0, 1])
        probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        for pos_class in (0, 1):
            def roc_auc_score(binary_labels, score):
                np.testing.assert_array_equal(binary_labels, (labels == pos_class).astype(np.int64))
                np.testing.assert_array_equal(score, probs[:, pos_class])
                return 1.0
            result = eval(expression, {"np": np, "labels": labels, "probs": probs,
                                       "pos_class": pos_class, "roc_auc_score": roc_auc_score})
            self.assertEqual(result, 1.0)


if __name__ == "__main__":
    unittest.main()
