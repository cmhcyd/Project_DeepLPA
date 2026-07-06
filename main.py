import argparse
import os
import numpy as np
import pandas as pd
import torch
import re
import torch.nn as nn
import uproot
import matplotlib.pyplot as plt
import traceback
import json
import time
from utils.utils import _parse_name_list, get_files, build_dataset_kwargs, move_input_to_device, _unwrap_dataset, enable_train_hook, check_model_eval_state, _save_pred_to_outdir
from utils.plot import plot_training_metrics
from utils.saveroot import save_split_root, save_merged_root
from utils.dataset import ParticleDataset
from utils.analyzeimportance import analyze_feature_importance
from utils.analyze_correlation import analyze_feature_correlation
from utils.analyze_correlation_target import analyze_correlation_with_target
from utils.datacheck import check_data_quality
from captum.attr import IntegratedGradients
from pathlib import Path
from scipy.stats import ks_2samp
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional



_HAS_SKLEARN = True

def import_class(class_path: str):
    """Import a class from a fully qualified string, e.g. 'model.MLP.MLP'."""
    module_path, class_name = class_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)

def parse_args():
    parser = argparse.ArgumentParser()
    # Data and tree
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing train/test subdirectories")
    parser.add_argument("--tree_name", type=str, default="Tree1", help="ROOT tree name, default Tree1")
    # Output ROOT
    parser.add_argument("--keep_branches", type=str, default="ALL",
                        help='Keep original branches. "ALL" keeps all; can also pass comma-separated branch names')
    # Input features
    parser.add_argument("--scalar_branches", type=str, default="",
                        help="Comma-separated scalar branch names, e.g. mass,energy,chisq")
    parser.add_argument("--array_spec", type=str, default="",
                        help="Array branches and lengths, format like: [tracks_pt,64],[tracks_eta,64]")
    parser.add_argument("--pad_value", type=float, default=0.0, help="Value to pad arrays when length is less than specified")
    # Labels and task
    parser.add_argument("--task", type=str, choices=["classification", "regression"], default="classification")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes for classification or output dimension for regression")
    # Classification
    parser.add_argument("--pos_class", type=int, default=0, help="Positive class index for AUC and efficiency/FPR")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for TPR/FPR calculation")
    parser.add_argument("--target_branches", type=str, default="", help="Comma-separated label branch names")
    # Regression
    parser.add_argument("--reg_pred_name", type=str, default="pred", help="Prefix for regression prediction column names; for multi-dim will be <prefix>_1, <prefix>_2 ...")
    parser.add_argument("--reg_target_names", type=str, default="", help="List of target parameter names for regression; format like [X,Y,Z] or 'X,Y,Z'")
    # Model
    parser.add_argument("--model_class", type=str, required=True,
                        help="Fully qualified class name, e.g. 'model.MLP.MLP' or 'model.DRNmodfied.DRNmod'")
    parser.add_argument("--model_args", type=str, default="{}",
                        help="JSON string of additional arguments for the model constructor, e.g. '{\"width\":256, \"depth\":5, \"dropout\":0.25}'")
    # Loss
    parser.add_argument("--loss_class", type=str, required=True,
                        help="Fully qualified loss class name, e.g. 'model.focalloss.FocalLoss' or 'torch.nn.CrossEntropyLoss'")
    parser.add_argument("--loss_args", type=str, default="{}",
                        help="JSON string of additional arguments for the loss constructor, e.g. '{\"signal_class\":0, \"alpha\":0.5, \"gamma\":2.0}'")
    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda", help="cpu/cuda")
    # Early stopping
    parser.add_argument("--early_stop_patience", type=int, default=0,
                    help="Number of epochs with no improvement after which training stops. 0 = disable.")
    parser.add_argument("--early_stop_metric", type=str, default="loss",
                    help="Metric to trigger the early stopping, loss or AUC")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0,
                    help="Minimum change in monitored metric to qualify as improvement.")
    # Evaluation/split
    parser.add_argument("--val_ratio", type=float, default=0.5, help="Validation ratio from training set")
    parser.add_argument("--split_seed", type=int, default=42, help="Random split seed")
    # Other
    parser.add_argument("--mode", type=str, choices=["train", "infer"], default="train")
    parser.add_argument("--ckpt", type=str, default="drn_ckpt.pt")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize features")
    parser.add_argument("--plot_metrics", action="store_true", help="Save training metrics plot")
    parser.add_argument("--analyze_feature_importance", action="store_true", help="Analyze feature importance after training")
    parser.add_argument("--fi_samples", type=int, default=1000, help="Number of random samples for feature importance")
    parser.add_argument("--copy_extra_trees", type=str, default="", help="Extra tree names to copy to output, comma-separated, e.g. Tree1,Tree2")
    parser.add_argument("--analyze_feature_correlation", action="store_true", help="Analyze feature correlations and save heatmap/CSV")
    parser.add_argument("--corr_max_features", type=int, default=None, help="Max number of features to include in correlation analysis (by variance)")
    parser.add_argument("--analyze_target_correlation", action="store_true", help="Analyze correlation between model features and external target branches")
    parser.add_argument("--target_corr_branches", type=str, default="", help="Comma-separated scalar branch names not used as model inputs but to compute correlation with input features, e.g. mass,energy")
    # Inference output/save options (support no-label inference)
    parser.add_argument("--save_pred", type=str, default="", help="Path to save inference results (.npy), leave empty to skip")
    parser.add_argument("--pred_prob", action="store_true", help="Classification: output class probabilities instead of discrete class (discrete class also saved)")
    parser.add_argument("--store_root", action="store_true", help="Write inference (and optionally training) results to ROOT file")
    parser.add_argument("--store_root_per_file", action="store_true", help="Write inference results per input file to separate ROOT files")
    # Inference/normalization
    parser.add_argument("--scaler_file", type=str, default="", help="Normalization file (.npz) containing mean and std to load during inference (overrides runtime calculation)")
    parser.add_argument("--save_scaler", type=str, default="", help="Save computed normalization statistics to this path (.npz) during training")
    return parser.parse_args()

def get_datasets_and_loaders(args):
    reg_target_names_list = _parse_name_list(getattr(args, "reg_target_names", "") or "")
    extra_scalar_branches = _parse_name_list(getattr(args, "target_corr_branches", "") or "")

    if getattr(args, "scaler_file", ""):
        sf = args.scaler_file
        if os.path.exists(sf):
            try:
                data = np.load(sf)
                if "mean" in data and "std" in data:
                    norm_stats = {"mean": data["mean"], "std": data["std"]}
                    print(f"[INFO] Loaded norm_stats from scaler_file: {sf}")
                else:
                    print(f"[WARN] scaler_file {sf} does not contain mean/std keys, ignoring.")
            except Exception as e:
                print(f"[WARN] Failed to load scaler_file {sf}: {e}")
        else:
            print(f"[WARN] Specified scaler_file does not exist: {sf}")
    else:
        norm_stats = None

    train_files = get_files(args.data_dir, "train")
    test_files = get_files(args.data_dir, "infer")
    
    train_dataset = None
    train_full_loader_for_save = None
    train_loader = None
    val_loader = None
    norm_stats = None
    
    if len(train_files) > 0:
        train_kwargs = build_dataset_kwargs(args, norm_stats=None, reg_target_names_list=reg_target_names_list)
        train_kwargs["extra_scalar_branches"] = extra_scalar_branches
        train_dataset = ParticleDataset(root_files=train_files, **train_kwargs)
        
        if args.normalize and len(train_dataset) > 0:
            norm_stats = None
            if len(train_dataset) > 0 and args.normalize:
                if hasattr(train_dataset, "norm_stats") and train_dataset.norm_stats:
                    ns = train_dataset.norm_stats
                    if isinstance(ns, dict) and "mean" in ns and "std" in ns:
                        norm_stats = {"mean": ns["mean"], "std": ns["std"]}
                if norm_stats is None:
                    try:
                        X = getattr(train_dataset, "X", None)
                        if X is None:
                            raise AttributeError("train_dataset has no X attribute, cannot compute mean/std.")
                        import numpy as _np
                        if hasattr(X, "numpy"):
                            X_np = X.numpy()
                        else:
                            X_np = _np.asarray(X)
                        mean = X_np.mean(axis=0, keepdims=True)
                        std = X_np.std(axis=0, keepdims=True) + 1e-8
                        norm_stats = {"mean": mean, "std": std}
                    except Exception as e:
                        print(f"[WARN] Failed to compute norm_stats from train_dataset.X: {e}")
                        norm_stats = None

                if norm_stats is not None:
                    train_kwargs = build_dataset_kwargs(args, norm_stats=norm_stats, reg_target_names_list=reg_target_names_list)
                    train_kwargs["extra_scalar_branches"] = extra_scalar_branches
                    train_dataset = ParticleDataset(root_files=train_files, **train_kwargs)
                    if getattr(args, "save_scaler", ""):
                        try:
                            np.savez(args.save_scaler, mean=norm_stats["mean"], std=norm_stats["std"])
                            print(f"[INFO] Saved norm_stats to {args.save_scaler}")
                        except Exception as e:
                            print(f"[WARN] Failed to save norm_stats to {args.save_scaler}: {e}")

        if args.mode == "train" and len(train_dataset) > 0:
            val_size = int(args.val_ratio * len(train_dataset))
            train_size = len(train_dataset) - val_size
            
            if _HAS_SKLEARN:
                if args.task == "classification" and train_dataset.has_target:
                    labels = train_dataset.y
                    if labels.ndim > 1 and labels.shape[1] > 1:
                        labels = np.argmax(labels, axis=1)
                    else:
                        labels = labels.flatten()
                    
                    train_indices, val_indices = train_test_split(
                        np.arange(len(train_dataset)), 
                        test_size=val_size, 
                        stratify=labels,
                        random_state=args.split_seed
                    )
                    train_subset = Subset(train_dataset, train_indices)
                    val_subset = Subset(train_dataset, val_indices)
                else:
                    train_indices, val_indices = train_test_split(
                        np.arange(len(train_dataset)), 
                        test_size=val_size, 
                        random_state=args.split_seed
                    )
                    train_subset = Subset(train_dataset, train_indices)
                    val_subset = Subset(train_dataset, val_indices)
            else:
                from torch.utils.data import random_split
                train_subset, val_subset = random_split(
                    train_dataset, 
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(args.split_seed)
                )
            
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=True)
        
        train_full_loader_for_save = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.num_workers, pin_memory=True)
    
    test_loader = None
    test_dataset = None
    
    if len(test_files) > 0:
        test_kwargs = build_dataset_kwargs(args, norm_stats=norm_stats, reg_target_names_list=reg_target_names_list if (args.task=="regression" and reg_target_names_list) else None)
        test_kwargs["extra_scalar_branches"] = extra_scalar_branches
        if args.task == "classification" and not args.target_branches:
            test_kwargs["target_branches"] = []
        test_dataset = ParticleDataset(root_files=test_files, **test_kwargs)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
    
    return train_dataset, train_full_loader_for_save, test_dataset, train_loader, val_loader, test_loader, None, None

def train_one_epoch(model, loader, optimizer, criterion, device, task):

    feature_names = None
    try:
        from utils.utils import _unwrap_dataset
        ds = _unwrap_dataset(loader.dataset)
        if hasattr(ds, 'get_feature_names'):
            feature_names = ds.get_feature_names()
    except Exception:
        pass

    model.train()
    total_loss = 0.0
    total, correct = 0, 0
    
    for batch_idx, batch in enumerate(loader):
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise RuntimeError("Training data must include labels (x, y).")
        
        x = move_input_to_device(x, device)
        y = y.to(device)

        if isinstance(x, dict):
            x_tab = x.get("x_tab")
        else:
            x_tab = x

        if isinstance(x_tab, torch.Tensor) and (torch.isnan(x_tab).any() or torch.isinf(x_tab).any()):
            nan_mask = torch.isnan(x_tab)
            inf_mask = torch.isinf(x_tab)
            problematic_mask = nan_mask | inf_mask
            
            nan_feat_indices = torch.where(nan_mask.any(dim=0))[0]
            inf_feat_indices = torch.where(inf_mask.any(dim=0))[0]
            
            print(f"Warning: Batch {batch_idx} input contains NaN or infinity")
            if len(nan_feat_indices) > 0:
                idx_str = nan_feat_indices.tolist()
                print(f"  Features with NaN: {idx_str}")
                if feature_names is not None:
                    name_str = [feature_names[i] for i in idx_str]
                    print(f"    Corresponding names: {name_str}")
                for feat_idx in nan_feat_indices[:10]:
                    nan_samples = torch.where(nan_mask[:, feat_idx])[0]
                    name = feature_names[feat_idx] if feature_names else str(feat_idx)
                    print(f"    Feature {name}: NaN at samples {nan_samples[:5].tolist()}")
            
            if len(inf_feat_indices) > 0:
                idx_str = inf_feat_indices.tolist()
                print(f"  Features with infinity: {idx_str}")
                if feature_names is not None:
                    name_str = [feature_names[i] for i in idx_str]
                    print(f"    Corresponding names: {name_str}")
                for feat_idx in inf_feat_indices[:10]:
                    inf_samples = torch.where(inf_mask[:, feat_idx])[0]
                    inf_values = x[inf_samples[:3], feat_idx]
                    name = feature_names[feat_idx] if feature_names else str(feat_idx)
                    print(f"    Feature {name}: infinity at samples {inf_samples[:3].tolist()}, values: {inf_values.tolist()}")

            print(f"Warning: Batch {batch_idx} input contains NaN/inf, fixing")
            if len(nan_feat_indices) > 0:
                print(f"  Fixed {nan_mask.sum().item()} NaN values")
            if len(inf_feat_indices) > 0:
                print(f"  Fixed {inf_mask.sum().item()} infinity values")
            
            with torch.no_grad():
                feature_means = torch.zeros(x_tab.shape[1], device=device)
                for feat_idx in range(x_tab.shape[1]):
                    feat_values = x_tab[:, feat_idx]
                    feature_means[feat_idx] = 0.0
                x_tab[problematic_mask] = feature_means.repeat(x_tab.shape[0], 1)[problematic_mask]
            if isinstance(x, dict):
                x["x_tab"] = x_tab
            else:
                x = x_tab
        
        optimizer.zero_grad()
        logits = model(x)
        
        if task == "classification":
            if y.ndim == 2 and y.size(-1) == 1:
                y_ce = y.view(-1).long()
            elif y.ndim == 1:
                y_ce = y.long()
            else:
                y_ce = y.argmax(dim=-1).long()
            
            if torch.isnan(y_ce).any() or torch.isinf(y_ce).any():
                nan_labels = torch.isnan(y_ce)
                inf_labels = torch.isinf(y_ce)
                
                print(f"Warning: Batch {batch_idx} labels contain NaN/inf, fixing")
                
                if len(y_ce) > 0:
                    replacement_label = 0
                    y_ce[nan_labels | inf_labels] = replacement_label
                else:
                    print(f"  Error: No valid labels, skipping batch")
                    continue
                
            loss = criterion(logits, y_ce)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Batch {batch_idx} loss is NaN/inf, skipping batch")
                continue
                
            preds = logits.argmax(dim=-1)
            correct += (preds == y_ce).sum().item()
            total += y.size(0)
        else:
            if torch.isnan(y).any() or torch.isinf(y).any():
                nan_mask = torch.isnan(y)
                inf_mask = torch.isinf(y)
                
                print(f"Warning: Batch {batch_idx} regression labels contain NaN/inf, fixing")
                
                valid_mask = ~(nan_mask | inf_mask)
                if valid_mask.any():
                    y_mean = y[valid_mask].mean()
                    y[nan_mask | inf_mask] = y_mean
                else:
                    print(f"  Error: No valid labels, skipping batch")
                    continue
            
            loss = criterion(logits, y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Batch {batch_idx} loss is NaN/inf, skipping batch")
                continue
        
        loss.backward()
        
        grad_problematic = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_nan = torch.isnan(param.grad).any()
                grad_inf = torch.isinf(param.grad).any()
                
                if grad_nan or grad_inf:
                    print(f"Warning: Gradient for {name} contains {'NaN' if grad_nan else ''}{' and ' if grad_nan and grad_inf else ''}{'inf' if grad_inf else ''}, fixing")
                    
                    param.grad[torch.isnan(param.grad)] = 0.0
                    param.grad[torch.isinf(param.grad)] = 0.0
                    grad_problematic = True
        
        if grad_problematic:
            print(f"  Fixed gradient issues, continuing training")
        
        optimizer.step()
        total_loss += loss.item() * y.size(0)
    
    if total == 0:
        return float('nan'), float('nan')
    
    avg_loss = total_loss / len(loader.dataset)
    acc = (correct / total) if (task == "classification" and total > 0) else None
    return avg_loss, acc

@torch.no_grad()
def evaluate(model, loader, criterion, device, task, pos_class=1, threshold=0.5):
    model.eval()
    total_loss = 0.0
    total, correct = 0, 0
    all_probs = []
    all_labels = []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise RuntimeError("Evaluation data must include labels (x, y).")
        x = move_input_to_device(x, device)
        y = y.to(device)

        if isinstance(x, dict):
            x_tab = x.get("x_tab")
        else:
            x_tab = x

        if isinstance(x_tab, torch.Tensor) and (torch.isnan(x_tab).any() or torch.isinf(x_tab).any()):
            nan_mask = torch.isnan(x_tab)
            inf_mask = torch.isinf(x_tab)
            problematic_mask = nan_mask | inf_mask
            print(f"Warning: Evaluation data contains NaN/Inf, fixing")
            with torch.no_grad():
                feature_means = torch.zeros(x_tab.shape[1], device=device)
                x_tab[problematic_mask] = feature_means.repeat(x_tab.shape[0], 1)[problematic_mask]
            if isinstance(x, dict):
                x["x_tab"] = x_tab
            else:
                x = x_tab

        logits = model(x)
        if task == "classification":
            if y.ndim == 2 and y.size(-1) == 1:
                y_ce = y.view(-1).long()
            elif y.ndim == 1:
                y_ce = y.long()
            else:
                y_ce = y.argmax(dim=-1).long()
            loss = criterion(logits, y_ce)
            preds = logits.argmax(dim=-1)
            correct += (preds == y_ce).sum().item()
            total += y.size(0)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(y_ce.detach().cpu().numpy())
        else:
            loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
    avg_loss = total_loss / len(loader.dataset)
    acc = (correct / total) if (task == "classification" and total > 0) else None
    metrics = dict(loss=avg_loss, acc=acc, auc=None, sig_eff=None, bkg_acc=None, bkg_rej=None, bkg_ratio=None, sig_num=None, bkg_num=None)
    if task == "classification" and len(all_probs) > 0:
        probs = np.concatenate(all_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        try:
            if probs.shape[1] == 2:
                auc = (1.0-roc_auc_score(labels, probs[:, pos_class]))
            else:
                auc = (1.0-roc_auc_score(labels, probs, multi_class='ovr', average='macro'))
        except Exception:
            auc = float("nan")
        metrics["auc"] = float(auc)
        if probs.shape[1] == 2:
            pos_prob = probs[:, pos_class]
            preds_pos = (pos_prob >= threshold).astype(np.int64)
            y_true = (labels == pos_class).astype(np.int64)
            TP = np.sum((preds_pos == 1) & (y_true == 1))
            FP = np.sum((preds_pos == 1) & (y_true == 0))
            TN = np.sum((preds_pos == 0) & (y_true == 0))
            FN = np.sum((preds_pos == 0) & (y_true == 1))
            P = max(1, TP + FN)
            N = max(1, TN + FP)
            tpr = TP / P
            fpr = FP / N
            bkgt = TP / (TP + FP)
            metrics["sig_eff"] = float(tpr)
            metrics["bkg_acc"] = float(fpr)
            metrics["bkg_rej"] = float(1.0 - fpr)
            metrics["bkg_ratio"] = float(1.0 - bkgt)
            metrics["sig_num"] = float(TP)
            metrics["bkg_num"] = float(FP)
    return metrics

@torch.no_grad()
def infer_predict(model, loader, device, task, pred_prob=False):
    model.eval()
    preds = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        elif isinstance(batch, dict):
            x = batch.get("x", batch.get("features"))
        else:
            x = batch
        x = move_input_to_device(x, device)
        logits = model(x)
        if task == "classification":
            if pred_prob:
                out = torch.softmax(logits, dim=-1)
            else:
                out = torch.argmax(logits, dim=-1)
        else:
            out = logits
        preds.append(out.detach().cpu().numpy())
    if not preds:
        return np.zeros((0,))
    return np.concatenate(preds, axis=0)

@torch.no_grad()
def collect_predictions_from_loader(model, loader, device, task, force_save_prob=True):
    model.eval()
    y_list = []
    pred_class_list = []
    pred_prob_list = []
    pred_value_list = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x = batch[0]
                y = batch[1]
            elif isinstance(batch, dict):
                x = batch.get("x", batch.get("features"))
                y = batch.get("y", None)
            else:
                x = batch
                y = None

            x = x.to(device)
            if y is not None:
                if isinstance(y, torch.Tensor):
                    y_np = y.detach().cpu().numpy()
                else:
                    y_np = np.asarray(y)
                if y_np.ndim == 2 and y_np.shape[1] == 1:
                    y_np = y_np.reshape(-1)
                y_list.append(y_np)

            logits = model(x)
            if task == "classification":
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                if probs.ndim == 1:
                    probs = probs.reshape(-1, 1)
                if probs.shape[1] == 1:
                    cls = (probs[:, 0] >= 0.5).astype(np.int64)
                else:
                    cls = np.argmax(probs, axis=1).astype(np.int64)
                pred_class_list.append(cls)
                if force_save_prob:
                    pred_prob_list.append(probs)
            else:
                vals = logits.detach().cpu().numpy()
                if vals.ndim == 2 and vals.shape[1] == 1:
                    vals = vals.reshape(-1)
                pred_value_list.append(vals)
    y_true_all = np.concatenate(y_list, axis=0) if y_list else None
    pred_class_all = np.concatenate(pred_class_list, axis=0) if pred_class_list else None
    pred_prob_all = np.concatenate(pred_prob_list, axis=0) if pred_prob_list else None
    pred_value_all = np.concatenate(pred_value_list, axis=0) if pred_value_list else None

    return y_true_all, pred_class_all, pred_prob_all, pred_value_all

def save_predictions_for_loader(model, loader, args, device, task, tag: str, force_save_prob=True):
    if loader is None:
        print(f"[WARN] No loader to save for {tag}, skipping.")
        return

    reg_target_names_list = _parse_name_list(getattr(args, "reg_target_names", "") or "")
    reg_pred_name = getattr(args, "reg_pred_name", "pred")

    print(f"[INFO] Collecting and saving predictions for {tag}...")
    y_true_all, pred_class_all, pred_prob_all, pred_value_all = collect_predictions_from_loader(
        model, loader, device, task, force_save_prob=force_save_prob
    )

    if tag == "train":
        y_true_all = None

    ds_for_meta = _unwrap_dataset(loader.dataset)

    base, ext = os.path.splitext(args.save_pred)
    if ext == "":
        ext = ".root"
    out_root = f"{base}_{tag}{ext}"

    if args.store_root:
        if getattr(args, "store_root_per_file", False):
            written_paths = save_split_root(
                output_root=args.save_pred,
                tree_name=args.tree_name,
                dataset=ds_for_meta,
                task=task,
                y_true=y_true_all,
                pred_class=pred_class_all if (task == "classification" and pred_class_all is not None) else None,
                pred_prob=pred_prob_all if (task == "classification" and pred_prob_all is not None) else None,
                pred_value=pred_value_all if (task == "regression" and pred_value_all is not None) else None,
                train_mask=None,
                keep_branches_param=args.keep_branches,
                reg_pred_name=reg_pred_name,
                reg_target_names=reg_target_names_list if reg_target_names_list else None,
                extra_trees=args.copy_extra_trees
            )
            print(f"[INFO] Saved {tag} predictions per file to {len(written_paths)} ROOT files")
        else:
            save_merged_root(
                output_root=out_root,
                tree_name=args.tree_name,
                dataset=ds_for_meta,
                task=task,
                y_true=y_true_all,
                pred_class=pred_class_all if (task == "classification" and pred_class_all is not None) else None,
                pred_prob=pred_prob_all if (task == "classification" and pred_prob_all is not None) else None,
                pred_value=pred_value_all if (task == "regression" and pred_value_all is not None) else None,
                train_mask=None,
                keep_branches_param=args.keep_branches,
                reg_pred_name=reg_pred_name,
                reg_target_names=reg_target_names_list if reg_target_names_list else None,
                extra_trees=args.copy_extra_trees
            )
            print(f"[INFO] Saved {tag} predictions to {out_root}")

def main():
    print("DEBUG:0")
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("DEBUG:1")
    
    # Dynamically load loss
    loss_class = import_class(args.loss_class)
    loss_args = json.loads(args.loss_args)
    criterion = loss_class(**loss_args)
    print(f"Loaded loss: {args.loss_class} with args: {loss_args}")
    print("DEBUG:2")

    train_dataset, train_full_loader_for_save, test_dataset, train_loader, val_loader, test_loader, _, _ = get_datasets_and_loaders(args)
    print("DEBUG:3")

    if train_dataset is not None and len(train_dataset) > 0:
        check_data_quality(train_dataset, "Training set")

    ref_ds = train_dataset or test_dataset
    if ref_ds is None or len(ref_ds) == 0:
        print("No data found.")
        return
    in_dim = ref_ds.X.shape[1]
    print(f"Input dimension: {in_dim}")

    # Dynamically load model
    model_class = import_class(args.model_class)
    model_args = json.loads(args.model_args)
    model = model_class(in_dim=in_dim, num_classes=args.num_classes, task=args.task, **model_args)
    model = model.to(device)
    print(f"Loaded model: {args.model_class} with args: {model_args}")

    print(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")

    if train_loader is not None:
        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
                x_moved = move_input_to_device(x, device)
                if isinstance(x_moved, dict):
                    x_tab = x_moved.get("x_tab")
                    print(f"First batch input (tab) shape: {x_tab.shape}")
                    try:
                        print(f"First batch input (tab) range: [{x_tab.min().item():.6f}, {x_tab.max().item():.6f}]")
                    except Exception:
                        pass
                else:
                    print(f"First batch input shape: {x_moved.shape}")
                    try:
                        print(f"First batch input range: [{x_moved.min().item():.6f}, {x_moved.max().item():.6f}]")
                    except Exception:
                        pass

                with torch.no_grad():
                    logits = model(x_moved)
                    print(f"First batch output shape: {logits.shape}")
                    try:
                        print(f"First batch output range: [{logits.min().item():.6f}, {logits.max().item():.6f}]")
                    except Exception:
                        pass
                break
    print("DEBUG:4")

    if args.mode == "infer":
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        enable_train_hook(model)
        model.eval()
        check_model_eval_state(model, note="after loading ckpt")
        loader = test_loader if test_loader is not None else val_loader
        if loader is None:
            print("No validation/test data for evaluation.")
            return

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    x0 = batch[0]
                elif isinstance(batch, dict):
                    x0 = batch.get("x", batch.get("features"))
                else:
                    x0 = batch
                if isinstance(x0, dict):
                    x0_t = x0.get("x_tab")
                    if x0_t is None:
                        x0_t = torch.as_tensor(0)
                else:
                    if not isinstance(x0, torch.Tensor):
                        x0_t = torch.from_numpy(np.asarray(x0)).float()
                    else:
                        x0_t = x0.float()
                try:
                    x0_cpu = x0_t.cpu()
                    print(f"[DEBUG] first batch x shape={tuple(x0_cpu.shape)}, mean={float(x0_cpu.mean()):.6f}, std={float(x0_cpu.std()):.6f}")
                except Exception as e:
                    print(f"[WARN] Cannot print first batch stats: {e}")
                break

        if getattr(args, "scaler_file", ""):
            try:
                d = np.load(args.scaler_file)
                print(f"[DEBUG] Using scaler_file {args.scaler_file}: mean.shape={np.asarray(d['mean']).shape}, std.shape={np.asarray(d['std']).shape}")
            except Exception as e:
                print(f"[WARN] Cannot read scaler_file {args.scaler_file}: {e}")
        ds = _unwrap_dataset(loader.dataset)
        has_labels = getattr(ds, "has_target", False)
        if has_labels:
            metrics = evaluate(model, loader, criterion, device, args.task, args.pos_class, args.threshold)
            if args.task == "classification":
                print(f"Infer: loss={metrics['loss']:.4f}, acc={metrics['acc']:.4f}, auc={metrics['auc']:.4f}, "
                      f"sig_eff={metrics['sig_eff']:.4f}, bkg_rej={metrics['bkg_rej']:.4f}, bkg_acc={metrics['bkg_acc']:.4f}, bkg_ratio={metrics['bkg_ratio']:.4f}, Nsig={metrics['sig_num']:.4f}, Nbkg={metrics['bkg_num']:.4f},")
            else:
                print(f"Infer: loss={metrics['loss']:.4f}")

            y_list = []
            pred_class_list = []
            pred_prob_list = []
            pred_value_list = []

            model.eval()
            with torch.no_grad():
                for batch in loader:
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        x = batch[0]
                        y = batch[1]
                        x = move_input_to_device(x, device)
                        y_np = y.cpu().numpy()
                    elif isinstance(batch, dict):
                        x = batch.get("x", batch.get("features"))
                        x = move_input_to_device(x, device)
                        y = batch.get("y")
                        y_np = y.cpu().numpy() if y is not None else None
                    else:
                        x = move_input_to_device(batch, device)
                        y_np = None

                    logits = model(x)
                    if args.task == "classification":
                        if args.pred_prob:
                            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                            preds_cls = np.argmax(probs, axis=1)
                            pred_prob_list.append(probs)
                            pred_class_list.append(preds_cls)
                        else:
                            preds_cls = torch.argmax(logits, dim=-1).detach().cpu().numpy()
                            pred_class_list.append(preds_cls)
                            if args.store_root:
                                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                                pred_prob_list.append(probs)
                        if y_np is not None:
                            y_list.append(y_np.reshape(-1))
                    else:
                        vals = logits.detach().cpu().numpy()
                        pred_value_list.append(vals)
                        if y_np is not None:
                            y_list.append(y_np.reshape(-1))

            y_true_all = np.concatenate(y_list, axis=0) if y_list else None
            pred_class_all = np.concatenate(pred_class_list, axis=0) if pred_class_list else None
            pred_prob_all = np.concatenate(pred_prob_list, axis=0) if pred_prob_list else None
            pred_value_all = np.concatenate(pred_value_list, axis=0) if pred_value_list else None

            if args.save_pred:
                if args.task == "classification" and pred_class_all is not None:
                    np.save(args.save_pred, pred_class_all)
                    print(f"Infer: Saved predicted classes to {args.save_pred}, shape={pred_class_all.shape}")
                elif args.task == "regression" and pred_value_all is not None:
                    np.save(args.save_pred, pred_value_all)
                    print(f"Infer: Saved regression predictions to {args.save_pred}, shape={pred_value_all.shape}")
                else:
                    print(f"[WARN] No data to save (task={args.task}), skipping save to {args.save_pred}")

        else:
            preds = infer_predict(model, loader, device, args.task, pred_prob=args.pred_prob)
            if args.save_pred:
                np.save(args.save_pred, preds)
                print(f"Infer: Saved predictions to {args.save_pred}, shape={preds.shape}")
            else:
                print(f"Infer: Predictions done, shape={preds.shape} (use --save_pred to save; --pred_prob for probabilities)")

            y_true_all = None
            if args.pred_prob:
                pred_prob_all = preds
                pred_class_all = np.argmax(preds, axis=1) if preds.ndim > 1 else (preds > 0.5).astype(int)
            else:
                pred_prob_all = None
                pred_class_all = preds

            pred_value_all = None

        out_loader_for_save = loader
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_losses = []
        val_losses = []
        val_aucs = []
        signal_efficiencies = []
        background_rejections = []
        background_acceptances = []
        background_ratio = []
        background_num = []
        signal_num = []
        best_val = float("inf")
        best_auc = -1.0
        early_stop_counter = 0

        for epoch in range(1, args.epochs + 1):
            if train_loader is None:
                print("No training data.")
                break
            epoch_start = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, args.task)
            train_losses.append(tr_loss)

            if val_loader is not None and len(val_loader.dataset) > 0:
                metrics = evaluate(model, val_loader, criterion, device, args.task, args.pos_class, args.threshold)
                val_losses.append(metrics["loss"])
                val_aucs.append(metrics["auc"] if metrics["auc"] is not None else float("nan"))
                signal_efficiencies.append(metrics["sig_eff"] if metrics["sig_eff"] is not None else float("nan"))
                background_rejections.append(metrics["bkg_rej"] if metrics["bkg_rej"] is not None else float("nan"))
                background_acceptances.append(metrics["bkg_acc"] if metrics["bkg_acc"] is not None else float("nan"))
                background_ratio.append(metrics["bkg_ratio"] if metrics["bkg_ratio"] is not None else float("nan"))
                background_num.append(metrics["bkg_num"] if metrics["bkg_num"] is not None else float("nan"))
                signal_num.append(metrics["sig_num"] if metrics["sig_num"] is not None else float("nan"))
            else:
                val_losses.append(tr_loss)
                val_aucs.append(float("nan"))
                signal_efficiencies.append(float("nan"))
                background_rejections.append(float("nan"))
                background_acceptances.append(float("nan"))
                background_ratio.append(float("nan"))
                background_num.append(float("nan"))
                signal_num.append(float("nan"))

            if args.task == "classification":
                loss_improved = (val_losses[-1] < best_val - args.early_stop_min_delta)
                auc_improved = (np.isfinite(val_aucs[-1]) and val_aucs[-1] > best_auc + args.early_stop_min_delta)
                if args.early_stop_metric == "loss":
                    is_better = loss_improved
                elif args.early_stop_metric == "AUC":
                    is_better = auc_improved
                else:
                    is_better = loss_improved or auc_improved
            else:
                is_better = val_losses[-1] < best_val - args.early_stop_min_delta
            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start

            if args.task == "classification":
                print(f"Epoch {epoch}: train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f} | "
                      f"val_loss={val_losses[-1]:.4f}, val_auc={val_aucs[-1]:.4f}, "
                      f"Nsig={signal_num[-1]:.4f}, Nbkg={background_num[-1]:.4f}, "
                      f"sig_eff={signal_efficiencies[-1]:.4f}, bkg_rej={background_rejections[-1]:.4f}, bkg_ratio={background_ratio[-1]:.4f}"
                      f"time={epoch_time:.2f}s")
            else:
                print(f"Epoch {epoch}: train_loss={tr_loss:.4f} | val_loss={val_losses[-1]:.4f} | time={epoch_time:.2f}s")

            if is_better:
                best_val = min(best_val, val_losses[-1])
                if args.task == "classification" and np.isfinite(val_aucs[-1]):
                    best_auc = max(best_auc, val_aucs[-1])
                torch.save(model.state_dict(), args.ckpt)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if val_loader is not None and args.early_stop_patience > 0 and early_stop_counter >= args.early_stop_patience:
                    print(f"Early stopping triggered after {epoch} epochs (no improvement for {args.early_stop_patience} consecutive epochs).")
                    break

        print("Training finished, best weights saved to:", args.ckpt)
        if args.plot_metrics:
            out_dir_for_plots = _save_pred_to_outdir(args.save_pred)
            plot_training_metrics(
                train_losses, val_losses, val_aucs,
                signal_efficiencies, background_rejections, background_acceptances, background_ratio,
                out_dir=out_dir_for_plots
            )
            if out_dir_for_plots:
                print(f"Saved training curves to {out_dir_for_plots}/training_metrics.png")
            else:
                print("Saved training curves: training_metrics.png")

        if args.analyze_feature_importance and val_loader is not None:
            val_dataset = _unwrap_dataset(val_loader.dataset)
            feature_names = val_dataset.get_feature_names()
            X_val = val_dataset.X
            out_dir_for_plots = _save_pred_to_outdir(args.save_pred)
            fi = analyze_feature_importance(
                model, feature_names, X_val,
                device=device.type,
                n_samples=args.fi_samples,
                target_class=args.pos_class,
                out_dir=out_dir_for_plots
            )
            if fi is not None:
                print("[INFO] Feature importance analysis done.")
        if args.analyze_feature_correlation and val_loader is not None:
            val_dataset = _unwrap_dataset(val_loader.dataset)
            feature_names = val_dataset.get_feature_names()
            X_val = val_dataset.X
            out_dir_for_plots = _save_pred_to_outdir(args.save_pred)
            fi2 = analyze_feature_correlation(
                feature_names, X_val,
                n_samples=args.fi_samples,
                max_features=args.corr_max_features,
                method='pearson',   # or 'spearman'
                out_dir=out_dir_for_plots
            )
            if fi2 is not None:
                print("[INFO] Feature correlation analysis done.")
        if args.analyze_target_correlation and val_loader is not None:
            val_dataset = _unwrap_dataset(val_loader.dataset)
            if hasattr(val_dataset, 'extra_data') and val_dataset.extra_data is not None:
                feature_names = val_dataset.get_feature_names()
                X_val = val_dataset.X
                target_names = getattr(val_dataset, 'extra_scalar_branches', [])
                if not target_names:
                    print("[WARN] No extra_scalar_branches defined, skip target correlation.")
                else:
                    target_data = val_dataset.extra_data
                    out_dir_for_plots = _save_pred_to_outdir(args.save_pred)
                    fi3 = analyze_correlation_with_target(
                        feature_names, X_val,
                        target_names, target_data,
                        out_dir=out_dir_for_plots,
                        n_samples=args.fi_samples,
                        method='pearson'
                    )
                    if fi3 is not None:
                        print("[INFO] Target correlation analysis done.")
            else:
                print("[WARN] No extra branches loaded for target correlation. Please specify --target_corr_branches.")

    if args.store_root:
        if os.path.exists(args.ckpt):
            model.load_state_dict(torch.load(args.ckpt, map_location=device))
        else:
            print(f"[WARN] Checkpoint {args.ckpt} not found, using current model weights for prediction.")

        model.eval()
        save_predictions_for_loader(model, test_loader, args, device, args.task, tag="infer", force_save_prob=args.pred_prob)

        if args.mode == "train" and train_full_loader_for_save is not None:
            save_predictions_for_loader(model, train_full_loader_for_save, args, device, args.task, tag="train", force_save_prob=True)

if __name__ == "__main__":
    main()