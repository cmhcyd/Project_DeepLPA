import os
import re
from typing import List, Tuple, Any
import numpy as np
import traceback
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import uproot
from typing import List, Tuple, Dict, Optional

def get_files(dir_path: str, split: str) -> List[str]:
    base_path = os.path.join(dir_path, split)
    if not os.path.exists(base_path):
        return []
    all_files = []
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.endswith(".root"):
                all_files.append(os.path.join(root, f))
    return sorted(list(set(all_files)))

def _to_float_scalar(x: Any) -> float:
    if np.isscalar(x):
        try:
            return float(x)
        except Exception:
            return 0.0
    arr = np.array(x).reshape(-1)
    return float(arr[0]) if arr.size > 0 else 0.0

def parse_object_spec(spec: str) -> List[Tuple[str, List[str]]]:
    """
    Parse configuration string of the second mode:
      [sigma+,{mass,energy,chisq}];[lambda,{mass,energy,chisq,decaylength}]
    Returns list: [(branch_name, [param1, param2, ...]), ...]
    """
    if not spec or not spec.strip():
        return []
    groups = []
    pattern = re.compile(r'\[([^\],]+)\s*,\s*\{([^\}]*)\}\]')
    for match in pattern.finditer(spec):
        obj = match.group(1).strip()
        params_str = match.group(2).strip()
        if params_str == "":
            params = []
        else:
            params = [p.strip() for p in params_str.split(",") if p.strip()]
        groups.append((obj, params))
    if not groups:
        raise ValueError(f"object_spec parsing failed, check format: {spec}")
    return groups

def _parse_name_list(s: str):
    if not s:
        return []
    ss = s.strip()
    if ss.startswith("[") and ss.endswith("]"):
        ss = ss[1:-1]
    parts = [p.strip() for p in ss.split(",") if p.strip()]
    return parts

def parse_array_spec(spec: str) -> List[Tuple[str, int]]:
    if not spec:
        return []
    pattern = re.compile(r"\[\s*([^\],]+)\s*,\s*(\d+)\s*\]")
    result = []
    for name, n in pattern.findall(spec):
        name = name.strip()
        length = int(n)
        if length < 0:
            raise ValueError(f"Array length cannot be negative: [{name},{length}]")
        result.append((name, length))
    return result

def parse_scalar_branches(spec: str) -> List[str]:
    if not spec:
        return []
    return [s.strip() for s in spec.split(",") if s.strip()]

def enable_train_hook(model):
    if getattr(model, "_train_hooked", False):
        return
    orig_train = model.train
    def hooked_train(mode=True):
        print(f"[DEBUG-HOOK] model.train called with mode={mode}")
        traceback.print_stack(limit=5)
        return orig_train(mode)
    model.train = hooked_train
    model._orig_train = orig_train
    model._train_hooked = True

def disable_train_hook(model):
    if getattr(model, "_train_hooked", False):
        model.train = model._orig_train
        model._train_hooked = False

def check_model_eval_state(model, note=""):
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = "no-params"
    print(f"[DEBUG] {note} model.training={model.training}, device={dev}, grad_enabled={torch.is_grad_enabled()}")
    if model.training:
        print("[WARN] model.training==True -> forcing model.eval()")
        model.eval()

def move_input_to_device(x, device):
    if isinstance(x, dict):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return x
    
def _unwrap_dataset(ds):
    if isinstance(ds, Subset):
        return ds.dataset
    return ds

def _save_pred_to_outdir(save_pred: str):
    if not save_pred:
        return None
    p = Path(save_pred)
    return str(p.parent if p.suffix else p)

def _build_event_index_map_for_files(root_files, tree_name):
    """
    Build simple (file_idx, local_idx) sequential mapping, assuming ParticleDataset does not filter events.
    Returns np.array(file_idx), np.array(local_idx)
    """
    file_idx = []
    local_idx = []
    for fi, rf in enumerate(root_files):
        try:
            with uproot.open(rf) as f:
                tree = f[tree_name]
                n = tree.num_entries
        except Exception as e:
            print(f"[WARN] Failed to read entry count from {rf}: {e}")
            n = 0
        for i in range(n):
            file_idx.append(fi)
            local_idx.append(i)
    return np.asarray(file_idx, dtype=np.int32), np.asarray(local_idx, dtype=np.int32)

def build_dataset_kwargs(args, norm_stats: Optional[Dict] = None, reg_target_names_list: Optional[List[str]] = None) -> Dict:
    scalar_branches = parse_scalar_branches(args.scalar_branches)
    array_specs = parse_array_spec(args.array_spec)
    if args.task == "regression" and reg_target_names_list:
        target_branches = reg_target_names_list
    else:
        target_branches = parse_scalar_branches(args.target_branches)
    kwargs = dict(
        tree_name=args.tree_name,
        scalar_branches=scalar_branches,
        array_specs=array_specs,
        target_branches=target_branches,
        normalize=args.normalize,
        norm_stats=norm_stats,
        pad_value=args.pad_value,
    )
    return kwargs