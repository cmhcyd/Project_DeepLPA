from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
import torch
import numpy as np
import uproot

class ParticleDataset(Dataset):
    """
    Input specification:
    - scalar_branches: comma-separated list of scalar branch names
    - array_specs: list of tuples [(branch_name, length), ...] where each array branch
      is expanded to a fixed length. For each event, if actual length < length, pad with pad_value;
      if > length, truncate.
    - target_branches: label branch names (scalar or vector, if vector take first element)
    - normalize: whether to standardize features X (use provided norm_stats or compute on the fly)
    - extra_scalar_branches: scalar branches to load but NOT used as model features (stored in self.extra_data)
    """
    def __init__(self,
                 root_files: List[str],
                 tree_name: str = "infmom",
                 scalar_branches: Optional[List[str]] = None,
                 array_specs: Optional[List[Tuple[str, int]]] = None,
                 target_branches: Optional[List[str]] = None,
                 normalize: bool = False,
                 norm_stats: Optional[Dict[str, np.ndarray]] = None,
                 pad_value: float = 0.0,
                 extra_scalar_branches: Optional[List[str]] = None):
        super().__init__()
        self.root_files = list(root_files or [])
        self.tree_name = tree_name
        self.scalar_branches = [b for b in (scalar_branches or []) if b]
        seen = set()
        self.array_specs: List[Tuple[str, int]] = []
        for name, ln in (array_specs or []):
            name = name.strip()
            if name and name not in seen:
                self.array_specs.append((name, int(ln)))
                seen.add(name)
        self.target_branches = [t for t in (target_branches or []) if t]
        self.normalize = normalize
        self.norm_stats = norm_stats
        self.pad_value = float(pad_value)
        self.extra_scalar_branches = [b for b in (extra_scalar_branches or []) if b]
        self.file_event_counts: List[int] = []
        self.feature_names = self._make_feature_names()
        self.X, self.y, self.extra_data = self._load_all()
        self.X = torch.from_numpy(self.X).float()
        if self.y is not None:
            self.y = torch.from_numpy(self.y).float()
        if self.extra_data is not None:
            self.extra_data = torch.from_numpy(self.extra_data).float()

    def _make_feature_names(self) -> List[str]:
        """
        Generate feature_names list:
          - scalars: original name e.g. "mass"
          - arrays: "tracks_pt_0", "tracks_pt_1", ...
        Order: scalar_branches then array_specs, each array expanded by index.
        """
        names: List[str] = []
        for s in self.scalar_branches:
            names.append(s)
        for arr_name, length in self.array_specs:
            l = int(length)
            if l < 0:
                raise ValueError(f"Array length cannot be negative: [{arr_name},{l}]")
            for j in range(l):
                names.append(f"{arr_name}_{j}")
        return names

    def __len__(self):
        return self.X.shape[0]

    @property
    def has_target(self) -> bool:
        if hasattr(self, "target_branches") and self.target_branches:
            y = getattr(self, "y", None)
            if y is None:
                return True
            try:
                if hasattr(y, "shape"):
                    return y.shape[0] > 0
            except Exception:
                pass
            return False
        return False

    def __getitem__(self, idx: int):
        x = torch.as_tensor(self.X[idx], dtype=torch.float32)
        if self.has_target and self.y.shape[0] > idx:
            y = torch.as_tensor(self.y[idx], dtype=torch.float32)
            return x, y
        else:
            return x

    def get_feature_names(self) -> List[str]:
        return self.feature_names

    def get_event_index_map(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.event_file_idx, self.event_local_idx

    def _load_all(self) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        array_branch_names = [name for name, _ in self.array_specs]
        branches_to_read = set(self.scalar_branches + array_branch_names + self.target_branches + self.extra_scalar_branches)

        features: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        extra_list: List[np.ndarray] = []
        n_total = 0

        for file in self.root_files:
            with uproot.open(file) as f:
                if self.tree_name not in f:
                    raise RuntimeError(f"Tree '{self.tree_name}' not found in file {file}")
                tree = f[self.tree_name]

                available = set(tree.keys())
                read_list = [b for b in branches_to_read if b in available]
                missing = branches_to_read - set(read_list)
                if missing:
                    print(f"Warning: Missing branches in {file}: {sorted(list(missing))}")

                arrays = tree.arrays(read_list, library="np") if read_list else {}

                try:
                    if arrays:
                        n_events = len(next(iter(arrays.values())))
                    else:
                        n_events = int(getattr(tree, "num_entries", 0))
                except Exception:
                    n_events = 0

                for i in range(n_events):
                    feats: List[float] = []

                    for b in self.scalar_branches:
                        if b in arrays:
                            try:
                                val = arrays[b][i]
                                if isinstance(val, np.ndarray) and val.shape != ():
                                    v = float(np.asarray(val).reshape(-1)[0]) if val.size > 0 else 0.0
                                else:
                                    v = float(val)
                            except Exception:
                                v = 0.0
                        else:
                            v = 0.0
                        feats.append(v)

                    for b, length in self.array_specs:
                        l = int(length)
                        if b in arrays:
                            try:
                                raw = arrays[b][i]
                                vec = np.array(raw, dtype=np.float32).reshape(-1)
                            except Exception:
                                vec = np.asarray([], dtype=np.float32)
                        else:
                            vec = np.asarray([], dtype=np.float32)

                        if l == 0:
                            clipped = np.asarray([], dtype=np.float32)
                        else:
                            if vec.shape[0] >= l:
                                clipped = vec[:l]
                            else:
                                pad = np.full((l - vec.shape[0],), self.pad_value, dtype=np.float32)
                                clipped = np.concatenate([vec, pad], axis=0)
                        feats.extend(clipped.tolist())

                    features.append(np.asarray(feats, dtype=np.float32))

                    if self.target_branches:
                        tgt_vals: List[float] = []
                        for tb in self.target_branches:
                            if tb in arrays:
                                try:
                                    tv = arrays[tb][i]
                                    if isinstance(tv, np.ndarray):
                                        v = float(np.asarray(tv).reshape(-1)[0]) if tv.size > 0 else 0.0
                                    else:
                                        v = float(tv)
                                except Exception:
                                    v = 0.0
                            else:
                                v = 0.0
                            tgt_vals.append(v)
                        labels.append(np.asarray(tgt_vals, dtype=np.float32))

                    extra_vals: List[float] = []
                    for eb in self.extra_scalar_branches:
                        if eb in arrays:
                            try:
                                ev = arrays[eb][i]
                                if isinstance(ev, np.ndarray):
                                    v = float(np.asarray(ev).reshape(-1)[0]) if ev.size > 0 else 0.0
                                else:
                                    v = float(ev)
                            except Exception:
                                v = 0.0
                        else:
                            v = 0.0
                        extra_vals.append(v)
                    extra_list.append(np.asarray(extra_vals, dtype=np.float32))

                self.file_event_counts.append(n_events)
                n_total += n_events

        X = np.stack(features, axis=0) if features else np.zeros((0, len(self.feature_names)), dtype=np.float32)
        y = np.stack(labels, axis=0) if labels else None
        extra = np.stack(extra_list, axis=0) if extra_list else None
        if X.size > 0 and X.shape[1] != len(self.feature_names):
            raise RuntimeError(f"Constructed X column count ({X.shape[1]}) does not match feature_names length ({len(self.feature_names)}). "
                               "Check scalar_branches and array_specs against actual branches and lengths in ROOT files.")

        if self.normalize and X.size > 0:
            if self.norm_stats is not None and "mean" in self.norm_stats and "std" in self.norm_stats:
                mean = self.norm_stats["mean"]
                std = self.norm_stats["std"]
            else:
                mean = X.mean(axis=0, keepdims=True)
                std = X.std(axis=0, keepdims=True) + 1e-8
                self.norm_stats = {"mean": mean, "std": std}
            X = (X - mean) / std

        return X.astype(np.float32), (y.astype(np.float32) if y is not None else None), (extra.astype(np.float32) if extra is not None else None)