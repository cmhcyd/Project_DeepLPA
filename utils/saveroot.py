import numpy as np
from typing import List, Optional, Union
import uproot
import awkward as ak
import os

def _parse_keep_branches_param(param) -> Optional[List[str]]:
    """
    Parse keep_branches_param: None or "ALL" -> return None (keep all); comma-separated string -> list; already list -> return as is.
    """
    if param is None:
        return None
    if isinstance(param, list):
        return param
    s = str(param).strip()
    if s == "" or s.upper() == "ALL":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts if parts else None

def _ensure_dir_for_file(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _as_list(x: Optional[Union[str, List[str]]]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [s for s in (t.strip() for t in x) if s]
    if isinstance(x, str):
        return [s for s in (t.strip() for t in x.split(",")) if s]
    return []

def _concat_tree_across_files(root_files: List[str], tree_name: str) -> ak.Array:
    arrays_per_file = []
    fields_ref = None
    for rf in root_files:
        with uproot.open(rf) as fin:
            if tree_name not in fin:
                continue
            t = fin[tree_name]
            arr = t.arrays(library="ak")
            fields = tuple(arr.fields)
            if fields_ref is None:
                fields_ref = fields
            else:
                if fields != fields_ref:
                    raise ValueError(f"Extra tree '{tree_name}' has inconsistent branches across files. Reference file: {rf}")
            arrays_per_file.append(arr)
    if not arrays_per_file:
        return None
    if len(arrays_per_file) == 1:
        return arrays_per_file[0]
    return ak.concatenate(arrays_per_file, axis=0)

def save_merged_root(output_root: str, tree_name: str, dataset,
                     task: str,
                     y_true: np.ndarray = None,
                     pred_class: np.ndarray = None,
                     pred_prob: np.ndarray = None,
                     pred_value: np.ndarray = None,
                     train_mask: np.ndarray = None,
                     keep_branches_param: str = "ALL",
                     reg_pred_name: str = "pred",
                     reg_target_names: Optional[List[str]] = None,
                     extra_trees: Optional[Union[str, List[str]]] = None):
    """
    Write merged output: original branches of the main tree + model outputs into one ROOT file.
    Additional trees can be copied as separate trees.
    Requires dataset.root_files and dataset.file_event_counts with event order matching concatenation.
    """
    root_files = list(dataset.root_files)
    file_event_counts = list(dataset.file_event_counts)
    if len(root_files) != len(file_event_counts):
        raise ValueError("dataset.root_files and dataset.file_event_counts length mismatch")

    keep_all = (str(keep_branches_param).upper() == "ALL")
    kept_fields = None
    arrays_main_per_file = []
    for rf in root_files:
        with uproot.open(rf) as fin:
            if tree_name not in fin:
                raise KeyError(f"File {rf} does not contain tree '{tree_name}'")
            t = fin[tree_name]
            if keep_all:
                fields = tuple(t.keys())
            else:
                fields = tuple(s.strip() for s in str(keep_branches_param).split(",") if s.strip())
                missing = [f for f in fields if f not in t.keys()]
                if missing:
                    raise KeyError(f"File {rf} tree '{tree_name}' missing branches: {missing}")
            if kept_fields is None:
                kept_fields = fields
            else:
                if tuple(fields) != tuple(kept_fields):
                    raise ValueError(f"Main tree '{tree_name}' kept branches inconsistent across files. File: {rf}")
            arr = t.arrays(list(kept_fields), library="ak")
            arrays_main_per_file.append(arr)

    if len(arrays_main_per_file) == 1:
        merged_main = arrays_main_per_file[0]
    else:
        merged_main = ak.concatenate(arrays_main_per_file, axis=0)

    out_branches = {name: merged_main[name] for name in kept_fields}

    N = len(merged_main)
    def _check_len(x, name):
        if x is not None and len(x) != N:
            raise ValueError(f"{name} length {len(x)} does not match event count {N}")

    _check_len(y_true, "y_true")
    _check_len(pred_class, "pred_class")
    _check_len(pred_prob, "pred_prob")
    _check_len(pred_value, "pred_value")
    _check_len(train_mask, "train_mask")

    if y_true is not None:
        y = np.asarray(y_true)
        if y.ndim == 1:
            out_branches["y_true"] = y
        elif y.ndim == 2:
            D = y.shape[1]
            if reg_target_names and len(reg_target_names) == D:
                for i, nm in enumerate(reg_target_names):
                    out_branches[str(nm)] = y[:, i]
            else:
                for i in range(D):
                    out_branches[f"y_true_{i+1}"] = y[:, i]
        else:
            raise ValueError("y_true dimensionality not supported, must be [N] or [N,D]")

    if task == "classification":
        if pred_class is not None:
            out_branches["pred_class"] = np.asarray(pred_class)
        if pred_prob is not None:
            pp = np.asarray(pred_prob)
            if pp.ndim == 1:
                out_branches["pred_prob"] = pp
            elif pp.ndim == 2:
                C = pp.shape[1]
                for i in range(C):
                    out_branches[f"pred_prob_{i+1}"] = pp[:, i]
            else:
                raise ValueError("pred_prob dimensionality not supported, must be [N] or [N,C]")
    elif task == "regression":
        if pred_value is not None:
            pv = np.asarray(pred_value)
            if pv.ndim == 1:
                out_branches[reg_pred_name] = pv
            elif pv.ndim == 2:
                Dp = pv.shape[1]
                if Dp == 1:
                    out_branches[reg_pred_name] = pv.reshape(-1)
                else:
                    for i in range(Dp):
                        out_branches[f"{reg_pred_name}_{i+1}"] = pv[:, i]
            else:
                raise ValueError("pred_value dimensionality not supported, must be [N] or [N,D]")
    if train_mask is not None:
        out_branches["is_train"] = np.asarray(train_mask).astype(np.int8)

    base, ext = os.path.splitext(output_root)
    if ext == "":
        ext = ".root"
    out_path = f"{base}{ext}"
    _ensure_dir_for_file(out_path)

    extra_tree_list = _as_list(extra_trees)
    with uproot.recreate(out_path) as fout:
        fout[tree_name] = out_branches
        for et in extra_tree_list:
            if et == tree_name:
                continue
            merged_extra = _concat_tree_across_files(root_files, et)
            if merged_extra is None:
                print(f"[WARN] Extra tree '{et}' not found in any input file, skipping")
                continue
            if hasattr(merged_extra, "fields"):
                fout[et] = {name: merged_extra[name] for name in merged_extra.fields}
            else:
                fout[et] = merged_extra
    print(f"[OK] Written merged ROOT: {out_path}")
    return out_path

def save_split_root(output_root: str, tree_name: str, dataset,
                    task: str,
                    y_true=None,
                    pred_class=None,
                    pred_prob=None,
                    pred_value=None,
                    train_mask=None,
                    keep_branches_param="ALL",
                    reg_pred_name: str = "pred",
                    reg_target_names: Optional[List[str]] = None,
                    extra_trees: Optional[Union[str, List[str]]] = None):
    """
    Write per input file: each output file contains the main tree (original branches + model outputs) and specified extra trees (copied as is).
    - output_root: if a directory or does not end with .root, output to that directory; if a .root path, use its parent directory.
    - output filename: <input_filename_stem>_pred.root
    """
    import pathlib

    root_files = list(dataset.root_files)
    file_event_counts = list(dataset.file_event_counts)
    if len(root_files) != len(file_event_counts):
        raise ValueError("dataset.root_files and dataset.file_event_counts length mismatch")

    out_root = output_root or ""
    out_path = pathlib.Path(out_root)
    if out_root.endswith(".root"):
        out_dir = out_path.parent
    else:
        out_dir = out_path if (out_root and out_path.suffix != ".root") else pathlib.Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_tree_list = _as_list(extra_trees)

    written = []
    start = 0
    for rf, n in zip(root_files, file_event_counts):
        end = start + n
        y_true_i = None if y_true is None else np.asarray(y_true)[start:end]
        pred_class_i = None if pred_class is None else np.asarray(pred_class)[start:end]
        pred_prob_i = None if pred_prob is None else np.asarray(pred_prob)[start:end]
        pred_value_i = None if pred_value is None else np.asarray(pred_value)[start:end]
        train_mask_i = None if train_mask is None else np.asarray(train_mask)[start:end]

        with uproot.open(rf) as fin:
            if tree_name not in fin:
                raise KeyError(f"File {rf} does not contain tree '{tree_name}'")
            t = fin[tree_name]

            if str(keep_branches_param).upper() == "ALL":
                keep_fields = list(t.keys())
            else:
                keep_fields = [s.strip() for s in str(keep_branches_param).split(",") if s.strip()]
                missing = [f for f in keep_fields if f not in t.keys()]
                if missing:
                    raise KeyError(f"File {rf} tree '{tree_name}' missing branches: {missing}")

            arr_main = t.arrays(keep_fields, library="ak")
            out_branches = {name: arr_main[name] for name in keep_fields}

            if y_true_i is not None:
                if y_true_i.ndim == 1:
                    out_branches["y_true"] = y_true_i
                elif y_true_i.ndim == 2:
                    D = y_true_i.shape[1]
                    if reg_target_names and len(reg_target_names) == D:
                        for i, nm in enumerate(reg_target_names):
                            out_branches[str(nm)] = y_true_i[:, i]
                    else:
                        for i in range(D):
                            out_branches[f"y_true_{i+1}"] = y_true_i[:, i]
                else:
                    raise ValueError("y_true dimensionality not supported, must be [N] or [N,D]")

            if task == "classification":
                if pred_class_i is not None:
                    out_branches["pred_class"] = pred_class_i
                if pred_prob_i is not None:
                    if pred_prob_i.ndim == 1:
                        out_branches["pred_prob"] = pred_prob_i
                    elif pred_prob_i.ndim == 2:
                        C = pred_prob_i.shape[1]
                        for i in range(C):
                            out_branches[f"pred_prob_{i+1}"] = pred_prob_i[:, i]
                    else:
                        raise ValueError("pred_prob dimensionality not supported, must be [N] or [N,C]")
            elif task == "regression":
                if pred_value_i is not None:
                    if pred_value_i.ndim == 1:
                        out_branches[reg_pred_name] = pred_value_i
                    elif pred_value_i.ndim == 2:
                        Dp = pred_value_i.shape[1]
                        if Dp == 1:
                            out_branches[reg_pred_name] = pred_value_i.reshape(-1)
                        else:
                            for i in range(Dp):
                                out_branches[f"{reg_pred_name}_{i+1}"] = pred_value_i[:, i]
                    else:
                        raise ValueError("pred_value dimensionality not supported, must be [N] or [N,D]")
            if train_mask_i is not None:
                out_branches["is_train"] = train_mask_i.astype(np.int8)

            stem = pathlib.Path(rf).stem
            out_file = out_dir / f"{stem}_pred.root"
            _ensure_dir_for_file(str(out_file))

            with uproot.recreate(str(out_file)) as fout:
                fout[tree_name] = out_branches
                for et in extra_tree_list:
                    if et == tree_name:
                        continue
                    if et not in fin:
                        print(f"[WARN] File {rf} does not have extra tree '{et}', skipping")
                        continue
                    arr_et = fin[et].arrays(library="ak")
                    if hasattr(arr_et, "fields"):
                        branch_dict = {name: arr_et[name] for name in arr_et.fields}
                    elif isinstance(arr_et, dict):
                        branch_dict = dict(arr_et)
                    else:
                        try:
                            branch_dict = dict(arr_et)
                        except Exception:
                            branch_dict = arr_et
                    fout[et] = branch_dict

            written.append(str(out_file))
        start = end

    print(f"[OK] Written per-file ROOT files: {len(written)}")
    return written