import ast
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import uproot
import awkward as ak

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.FOMloss import FOMLoss
from model.DRNmodfied import DRNmod
from utils.dataset import ParticleDataset, normalize_features, validate_classification_targets
from utils.saveroot import save_merged_root, save_split_root
from utils.utils import _parse_name_list, _unwrap_dataset, build_dataset_kwargs, get_files
from utils.datacheck import check_data_quality
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# Exercise the actual loader/CLI without importing optional analysis modules.
source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
             and node.name in ("get_datasets_and_loaders", "parse_args",
                               "collect_predictions_from_loader", "save_predictions_for_loader")]
import argparse
import os
namespace = dict(globals(), _HAS_SKLEARN=True)
exec(compile(ast.Module(body=functions, type_ignores=[]), "main.py", "exec"), namespace)
get_loaders = namespace["get_datasets_and_loaders"]


def write_events(path, values, labels=None, arrays=False):
    data = {"feature": np.asarray(values, dtype=np.float32)}
    if labels is not None:
        data.update(is_sig=np.asarray(labels == 0, dtype=np.int32),
                    is_bkg=np.asarray(labels == 1, dtype=np.int32))
    if arrays:
        data["tracks"] = ak.Array([[1.0, 2.0], [], [3.0], [4.0, 5.0, 6.0]])
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as output:
        output.mktree("Tree1", data)
        output.mktree("Metadata", {"run": np.array([7], dtype=np.int32)})


def make_args(directory, **kwargs):
    values = dict(data_dir=str(directory), mode="train", tree_name="Tree1",
                  scalar_branches="feature", array_spec="", pad_value=0,
                  target_branches="is_sig,is_bkg", reg_target_names="",
                  target_corr_branches="", num_classes=2, task="classification",
                  normalize=False, scaler_file="", save_scaler="", val_ratio=0.25,
                  split_seed=42, batch_size=4, num_workers=0)
    values.update(kwargs)
    return SimpleNamespace(**values)


class PreprocessingTests(unittest.TestCase):
    def test_inference_targets_are_optional_but_training_targets_are_required(self):
        for task in ("classification", "regression"):
            for mode in ("train", "infer"):
                with self.subTest(task=task, mode=mode), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    write_events(root / "train" / "sample.root", np.arange(20), np.arange(20) % 2)
                    # Labels remain available for metrics when every inference file has them.
                    write_events(root / "infer" / "a.root", [1, 2], np.array([0, 1]))
                    args = make_args(root, task=task, mode=mode)
                    self.assertTrue(get_loaders(args)[2].has_target)
                    # A mixed labeled/unlabeled sample must not build a partial target array.
                    write_events(root / "infer" / "b.root", [3, 4])
                    loaded = get_loaders(args)
                    self.assertFalse(loaded[2].has_target)
                    self.assertIsNone(loaded[2].y)
                    self.assertEqual(len(loaded[2]), 4)
                    self.assertIsInstance(next(iter(loaded[5])), torch.Tensor)
                    # Partially available target branches also mean unlabeled inference.
                    with uproot.recreate(root / "infer" / "b.root") as output:
                        output.mktree("Tree1", {"feature": np.array([3, 4], dtype=np.float32),
                                                "is_sig": np.array([0, 1], dtype=np.int32)})
                    self.assertFalse(get_loaders(args)[2].has_target)
                    with uproot.recreate(root / "infer" / "b.root") as output:
                        output.mktree("Tree1", {"other": np.array([3, 4], dtype=np.float32)})
                    with self.assertRaisesRegex(KeyError, "Required branches.*feature"):
                        get_loaders(args)
                    write_events(root / "train" / "sample.root", np.arange(20))
                    with self.assertRaisesRegex(KeyError, "Required branches"):
                        get_loaders(make_args(root, task=task))

    def test_scaler_fit_excludes_validation_and_is_reused_for_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.arange(20) % 2
            values = np.arange(20, dtype=np.float32)
            file = root / "train" / "sample.root"
            write_events(file, values, labels)
            raw = get_loaders(make_args(root))
            train_indices = raw[3].dataset.indices
            val_indices = raw[4].dataset.indices
            values[val_indices] += 1000
            write_events(file, values, labels)
            write_events(root / "infer" / "sample.root", [2, 4])
            scaler = root / "outputs" / "scaler.npz"
            loaded = get_loaders(make_args(root, normalize=True, save_scaler=str(scaler)))
            self.assertFalse(loaded[2].has_target)
            with np.load(scaler) as stats:
                np.testing.assert_allclose(stats["mean"], [[values[train_indices].mean()]])
                self.assertNotAlmostEqual(float(stats["mean"][0, 0]), float(values.mean()))
                expected = (np.array([[2], [4]]) - stats["mean"]) / stats["std"]
                np.testing.assert_allclose(loaded[2].X.numpy(), expected, rtol=1e-6)
            np.testing.assert_allclose(loaded[0].X[train_indices].mean(dim=0).numpy(), [0], atol=1e-6)
            # Same event processed alone must not be normalized against its test sample.
            write_events(root / "infer" / "sample.root", [2])
            standalone = get_loaders(make_args(root, mode="infer", target_branches="",
                                               normalize=False, scaler_file=str(scaler)))
            np.testing.assert_allclose(standalone[2].X.numpy()[0], loaded[2].X.numpy()[0])
            check_data_quality(standalone[0], "Unlabeled training files in inference mode")

    def test_external_scaler_takes_precedence_and_enables_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_events(root / "train" / "sample.root", np.arange(20), np.arange(20) % 2)
            scaler = root / "scaler.npz"
            np.savez(scaler, mean=[[100.0]], std=[[2.0]])
            loaded = get_loaders(make_args(root, scaler_file=str(scaler)))
            np.testing.assert_allclose(loaded[0].X.numpy().ravel(), (np.arange(20) - 100) / 2)

    def test_normalized_inference_rejects_missing_or_invalid_scaler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "requires --scaler_file"):
                get_loaders(make_args(root, mode="infer", normalize=True))
            with self.assertRaises(FileNotFoundError):
                get_loaders(make_args(root, mode="infer", scaler_file=str(root / "missing.npz")))
        for stats in ({"mean": [0, 1], "std": [1, 1]}, {"mean": [0], "std": [0]},
                      {"mean": [np.nan], "std": [1]}):
            with self.subTest(stats=stats), self.assertRaises(ValueError):
                normalize_features(np.ones((3, 1), dtype=np.float32), stats)

    def test_missing_required_branches_and_invalid_targets_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "sample.root"
            write_events(file, [1, 2])
            for kwargs in ({"scalar_branches": ["absent"]}, {"target_branches": ["is_sig"]}):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(KeyError, "Required branches"):
                    ParticleDataset([str(file)], tree_name="Tree1", **kwargs)
        for targets in (None, [[0, 0]], [[1, 1]], [[0.4, 0.6]], [[1, 0, 0]], [0, 2], [0.5], [np.nan]):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                validate_classification_targets(targets, 2)
        validate_classification_targets([[1, 0], [0, 1]], 2)
        validate_classification_targets([0, 1], 2)


class LossAndCLITests(unittest.TestCase):
    def test_fom_matches_documented_formula_and_has_finite_gradients(self):
        logits = torch.zeros((200, 2), requires_grad=True)
        targets = torch.cat([torch.zeros(100, dtype=torch.long), torch.ones(100, dtype=torch.long)])
        for normalized, expected in ((True, -0.05), (False, -5.0)):
            loss = FOMLoss(normalize_by_pos=normalized)(logits, targets)
            self.assertAlmostEqual(float(loss.detach()), expected, places=6)
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            logits.grad.zero_()
        no_signal = FOMLoss()(logits, torch.ones(200, dtype=torch.long))
        no_signal.backward()
        self.assertEqual(float(no_signal.detach()), 0.0)
        self.assertEqual(int(torch.count_nonzero(logits.grad)), 0)

    def test_early_stop_auc_accepts_both_cases_and_rejects_typos(self):
        required = ["main.py", "--data_dir", "/data", "--num_classes", "2",
                    "--model_class", "Model", "--loss_class", "Loss"]
        for metric in ("auc", "AUC", "AuC"):
            with patch.object(sys, "argv", required + ["--early_stop_metric", metric]):
                self.assertEqual(namespace["parse_args"]().early_stop_metric, "auc")
        with patch.object(sys, "argv", required + ["--early_stop_metric", "acc"]), self.assertRaises(SystemExit):
            namespace["parse_args"]()


class ROOTOutputTests(unittest.TestCase):
    def test_training_outputs_match_actual_split_and_preserve_predictions(self):
        for task in ("classification", "regression"):
            for per_file in (False, True):
                for sklearn_split in (False, True):
                    with self.subTest(task=task, per_file=per_file, sklearn_split=sklearn_split), tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        for index in range(2):
                            write_events(root / "train" / f"sample{index}.root",
                                         np.arange(4) + 4 * index, np.arange(4) % 2, arrays=True)
                        args = make_args(root, task=task, store_root=True,
                                         store_root_per_file=per_file, keep_branches="ALL",
                                         copy_extra_trees="Metadata", save_pred=str(root / "output" / "pred.root"))
                        with patch.dict(namespace, _HAS_SKLEARN=sklearn_split):
                            loaded = get_loaders(args)
                        dataset, full_loader, _, train_loader, val_loader = loaded[:5]
                        train_indices = np.asarray(train_loader.dataset.indices)
                        val_indices = np.asarray(val_loader.dataset.indices)
                        model = torch.nn.Linear(1, 2 if task == "classification" else 1)
                        collector = namespace["collect_predictions_from_loader"]
                        with patch.dict(namespace, collect_predictions_from_loader=unittest.mock.Mock(wraps=collector)):
                            namespace["save_predictions_for_loader"](
                                model, full_loader, args, torch.device("cpu"), task,
                                tag="train", train_indices=train_indices)
                            self.assertEqual(namespace["collect_predictions_from_loader"].call_count, 1)
                        train_mask = np.zeros(len(dataset), dtype=bool)
                        train_mask[train_indices] = True
                        np.testing.assert_array_equal(np.flatnonzero(~train_mask), np.sort(val_indices))
                        if per_file:
                            paths = [(root / "output" / f"sample{i}_pred.root",
                                      root / "output" / f"sample{i}_train_subset.root",
                                      root / "output" / f"sample{i}_test_subset.root") for i in range(2)]
                        else:
                            paths = [(root / "output" / "pred_train.root",
                                      root / "output" / "pred_train_subset.root",
                                      root / "output" / "pred_test_subset.root")]
                        for i, (full_path, train_path, test_path) in enumerate(paths):
                            mask = train_mask[i * 4:(i + 1) * 4] if per_file else train_mask
                            with uproot.open(full_path) as full:
                                original = full["Tree1"].arrays(library="ak")
                                expected_values = np.arange(i * 4, (i + 1) * 4) if per_file else np.arange(8)
                                np.testing.assert_array_equal(ak.to_numpy(original.feature), expected_values)
                                self.assertIn("pred_prob_1" if task == "classification" else "pred", original.fields)
                                for path, selection in ((train_path, mask), (test_path, ~mask)):
                                    with uproot.open(path) as subset:
                                        self.assertEqual(subset["Tree1"].classname, "TTree")
                                        actual = subset["Tree1"].arrays(library="ak")
                                        self.assertEqual(ak.to_list(actual), ak.to_list(original[selection]))
                                        self.assertEqual(subset["Metadata"].num_entries, 2 if not per_file else 1)

    def test_event_mask_rejects_wrong_shape_or_type(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.root"
            write_events(source, [0, 1, 2, 3])
            dataset = ParticleDataset([str(source)], tree_name="Tree1", scalar_branches=["feature"])
            for writer in (save_merged_root, save_split_root):
                for mask in (np.array([True]), np.array([1, 0, 1, 0])):
                    with self.subTest(writer=writer.__name__, mask=mask), self.assertRaisesRegex(ValueError, "event_mask"):
                        writer(str(root / "out.root"), "Tree1", dataset,
                               task="classification", event_mask=mask)

    def test_empty_subset_keeps_branch_schema_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.root"
            write_events(source, [0, 1, 2, 3], arrays=True)
            dataset = ParticleDataset([str(source)], tree_name="Tree1", scalar_branches=["feature"])
            kwargs = dict(task="classification", pred_prob=np.ones((4, 2)) / 2,
                          event_mask=np.zeros(4, dtype=bool), extra_trees="Metadata")
            merged = save_merged_root(str(root / "merged.root"), "Tree1", dataset, **kwargs)
            split = save_split_root(str(root / "split"), "Tree1", dataset, **kwargs)
            for path in [merged, *split]:
                with uproot.open(path) as output:
                    self.assertEqual(output["Tree1"].num_entries, 0)
                    self.assertIn("tracks", output["Tree1"].keys())
                    self.assertIn("pred_prob_1", output["Tree1"].keys())
                    self.assertEqual(output["Metadata"].num_entries, 1)

    def test_merged_and_per_file_outputs_preserve_ttrees_and_jagged_branches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.root"
            write_events(source, [1, 2, 3, 4], np.array([0, 1, 0, 1]), arrays=True)
            dataset = ParticleDataset([str(source)], tree_name="Tree1", scalar_branches=["feature"])
            kwargs = dict(task="classification", pred_class=np.array([0, 1, 0, 1]),
                          pred_prob=np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]),
                          train_mask=np.array([1, 0, 1, 0]), extra_trees="Metadata")
            merged = save_merged_root(str(root / "merged.root"), "Tree1", dataset, **kwargs)
            split = save_split_root(str(root / "split"), "Tree1", dataset, **kwargs)
            for path in [merged, *split]:
                with uproot.open(path) as output:
                    self.assertEqual(output["Tree1"].classname, "TTree")
                    self.assertEqual(output["Metadata"].classname, "TTree")
                    np.testing.assert_array_equal(output["Tree1"]["feature"].array(library="np"), [1, 2, 3, 4])
                    self.assertEqual(ak.to_list(output["Tree1"]["tracks"].array()), [[1, 2], [], [3], [4, 5, 6]])
                    np.testing.assert_allclose(output["Tree1"]["pred_prob_1"].array(library="np"), kwargs["pred_prob"][:, 0])


if __name__ == "__main__":
    unittest.main()
