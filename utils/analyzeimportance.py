import torch
from captum.attr import IntegratedGradients
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

def analyze_feature_importance(model, feature_names, X_val, device='cpu', n_samples=5000, batch_size=512, target_class=0, out_dir: str = None):
    model.to(device)
    model.eval()
    if not isinstance(X_val, torch.Tensor):
        X_tensor = torch.tensor(X_val, dtype=torch.float32, device=device, requires_grad=True)
    else:
        X_tensor = X_val.clone().detach().requires_grad_(True).to(device)
    if len(X_tensor) > n_samples:
        indices = torch.randperm(len(X_tensor))[:n_samples]
        X_sampled = X_tensor[indices]
    else:
        X_sampled = X_tensor
    dataset = TensorDataset(X_sampled)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ig = IntegratedGradients(model)
    total_attributions = []
    with torch.set_grad_enabled(True):
        for batch in dataloader:
            inputs = batch[0].to(device)
            inputs.requires_grad_(True)
            baseline = torch.zeros_like(inputs)
            attributions = ig.attribute(inputs, baselines=baseline, target=target_class)
            total_attributions.append(attributions.detach().cpu())
            del inputs, baseline, attributions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    avg_attr = torch.cat(total_attributions).mean(dim=0).numpy()
    sorted_idx = np.argsort(-np.abs(avg_attr))
    plt.figure(figsize=(12, 10))
    topk = min(20, len(feature_names))
    plt.barh(range(topk), avg_attr[sorted_idx[:topk]][::-1])
    plt.yticks(range(topk), [feature_names[i] for i in sorted_idx[:topk]][::-1])
    plt.xlabel("Integrated Gradients")
    plt.title("Top 20 important features", pad=20)
    plt.tight_layout()
    if out_dir:
        outp = Path(out_dir)
        if outp.suffix:
            outp = outp.parent
        outp.mkdir(parents=True, exist_ok=True)
        png_path = outp / "feature_importance.png"
        csv_path = outp / "feature_importance.csv"
    else:
        png_path = Path("feature_importance.png")
        csv_path = Path("feature_importance.csv")

    plt.savefig(str(png_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Feature importance figure saved to {png_path}")
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_scores = [float(avg_attr[i]) for i in sorted_idx]
    df = pd.DataFrame({
        'feature': sorted_names,
        'importance': sorted_scores
    })
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Feature importance CSV saved to {csv_path}")
    return {feature_names[i]: float(avg_attr[i]) for i in sorted_idx}