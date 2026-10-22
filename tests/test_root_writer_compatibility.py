from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.saveroot import save_merged_root, save_split_root


class ROOTWriterCompatibilityTests(unittest.TestCase):
    def check_outputs(self, event_mask):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for index in range(2):
                path = root / f"input{index}.root"
                values = {
                    "event_id": np.arange(4, dtype=np.int32) + index * 4,
                    "fixed": np.arange(12, dtype=np.float32).reshape(4, 3) + index,
                    "tracks": ak.Array([[1.0, 2.0], [], [3.0], [4.0, 5.0, 6.0]]),
                }
                with uproot.recreate(path) as output:
                    tree = output.mktree("Events", {"event_id": "int32", "fixed": "3 * float32",
                                                     "tracks": "var * float64"})
                    tree.extend(values)
                    meta = output.mktree("Metadata", {"run": "int32", "weights": "var * float64"})
                    meta.extend({"run": np.array([index], dtype=np.int32),
                                 "weights": ak.Array([[0.5, 1.0]])})
                inputs.append(str(path))
            dataset = SimpleNamespace(root_files=inputs, file_event_counts=[4, 4])
            probabilities = np.column_stack((np.arange(8) / 10, 1 - np.arange(8) / 10))
            train_mask = np.arange(8) % 2 == 0
            for task in ("classification", "regression"):
                kwargs = dict(task=task, train_mask=train_mask, event_mask=event_mask,
                              extra_trees="Metadata")
                if task == "classification":
                    kwargs.update(pred_class=np.argmax(probabilities, axis=1), pred_prob=probabilities)
                else:
                    kwargs.update(pred_value=np.arange(16, dtype=np.float32).reshape(8, 2))
                merged = save_merged_root(str(root / f"{task}.root"), "Events", dataset, **kwargs)
                per_file = save_split_root(str(root / task), "Events", dataset, **kwargs)
                for path, offset, count in [(merged, 0, 8), *[(p, i * 4, 4) for i, p in enumerate(per_file)]]:
                    with self.subTest(task=task, output=Path(path).name), uproot.open(path) as output:
                        selection = np.arange(offset, offset + count)
                        if event_mask is not None:
                            selection = selection[event_mask[offset:offset + count]]
                        tree = output["Events"]
                        self.assertEqual(tree.classname, "TTree")
                        self.assertEqual(tree.num_entries, len(selection))
                        actual = tree.arrays(library="ak")
                        np.testing.assert_array_equal(ak.to_numpy(actual.event_id), selection)
                        self.assertEqual(ak.to_numpy(actual.event_id).dtype, np.dtype("int32"))
                        expected_fixed = np.concatenate([
                            np.arange(12, dtype=np.float32).reshape(4, 3) + i for i in range(2)])
                        np.testing.assert_array_equal(ak.to_numpy(actual.fixed), expected_fixed[selection])
                        tracks = [[1.0, 2.0], [], [3.0], [4.0, 5.0, 6.0]] * 2
                        self.assertEqual(ak.to_list(actual.tracks), [tracks[i] for i in selection])
                        np.testing.assert_array_equal(ak.to_numpy(actual.is_train), train_mask[selection])
                        if task == "classification":
                            np.testing.assert_allclose(ak.to_numpy(actual.pred_prob_1), probabilities[selection, 0])
                            np.testing.assert_array_equal(ak.to_numpy(actual.pred_class),
                                                          np.argmax(probabilities[selection], axis=1))
                        else:
                            np.testing.assert_array_equal(ak.to_numpy(actual.pred_1), selection * 2)
                            np.testing.assert_array_equal(ak.to_numpy(actual.pred_2), selection * 2 + 1)
                        self.assertEqual(output["Metadata"].classname, "TTree")
                        self.assertEqual(output["Metadata"].num_entries, 2 if path == merged else 1)
                        self.assertEqual(ak.to_list(output["Metadata"]["weights"].array()),
                                         [[0.5, 1.0]] * (2 if path == merged else 1))

    def test_full_output(self):
        self.check_outputs(None)

    def test_selected_events(self):
        self.check_outputs(np.array([True, False, True, False, False, True, False, True]))

    def test_empty_subset(self):
        self.check_outputs(np.zeros(8, dtype=bool))


if __name__ == "__main__":
    print(f"Testing ROOT writers with Uproot {uproot.__version__}")
    unittest.main()
