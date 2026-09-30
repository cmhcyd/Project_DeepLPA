import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from pathlib import Path

def plot_custom_feature(feature_name, y_true, y_pred, df, train_mask):
    try:
        if feature_name not in df.columns:
            print(f"Feature '{feature_name}' not found, skipping analysis")
            return
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        if train_mask is None:
            train_mask = np.zeros_like(y_true, dtype=bool)
        else:
            train_mask = np.asarray(train_mask).astype(bool).reshape(-1)

        tp_mask = (y_true == 1) & (y_pred == 1)
        fp_mask = (y_true == 0) & (y_pred == 1)
        tn_mask = (y_true == 0) & (y_pred == 0)
        fn_mask = (y_true == 1) & (y_pred == 0)

        plt.figure(figsize=(18, 6))
        plt.suptitle(f"Feature Analysis: {feature_name}", y=1.01)

        plt.subplot(1, 3, 1)
        plt.hist(df.loc[y_true == 1, feature_name], bins=150, density=True, alpha=0.6,
                 label=f'Signal ({int(sum(y_true==1)):,})')
        plt.hist(df.loc[y_true == 0, feature_name], bins=150, density=True, alpha=0.6,
                 label=f'Background ({int(sum(y_true==0)):,})')
        plt.xlabel(feature_name); plt.ylabel('Normalized Event number'); plt.legend(); plt.title('True Distribution')

        plt.subplot(1, 3, 2)
        plt.hist(df.loc[y_pred == 1, feature_name], bins=150, density=True, alpha=0.5,
                 label=f'Pred Signal ({int(sum(y_pred==1)):,})')
        plt.hist(df.loc[y_pred == 0, feature_name], bins=150, density=True, alpha=0.5,
                 label=f'Pred Background ({int(sum(y_pred==0)):,})')
        plt.xlabel(feature_name); plt.ylabel('Normalized Event number'); plt.legend(); plt.title('Predicted Distribution')

        plt.subplot(1, 3, 3)
        if sum(tp_mask) > 0:
            plt.hist(df.loc[tp_mask, feature_name], bins=150, alpha=0.6, color='green',
                     label=f'Signal ({int(sum(tp_mask)):,})')
        if sum(fp_mask) > 0:
            plt.hist(df.loc[fp_mask, feature_name], bins=150, alpha=0.6, color='red',
                     label=f'Background ({int(sum(fp_mask)):,})')
        plt.xlabel(feature_name); plt.ylabel('Event number'); plt.legend()
        sig_eff = (sum(tp_mask) / max(1, sum(y_true == 1))) if (sum(y_true == 1) > 0) else 0.0
        denom = (sum(tp_mask) + sum(fp_mask))
        bkg_ratio = (sum(fp_mask) / denom) if denom > 0 else 0.0
        plt.title(f'Predicted Signal Composition\n(Signal Eff={sig_eff:.1%}, BKG ratio={bkg_ratio:.1%})')

        train_df = df[train_mask]
        val_df = df[~train_mask]
        train_pred_split = y_pred[train_mask]
        val_pred_split = y_pred[~train_mask]
        train_signal = train_df.loc[train_pred_split == 1, feature_name].dropna()
        val_signal = val_df.loc[val_pred_split == 1, feature_name].dropna()
        train_bkg = train_df.loc[train_pred_split == 0, feature_name].dropna()
        val_bkg = val_df.loc[val_pred_split == 0, feature_name].dropna()

        ks_results = []
        for (data1, data2, name) in [
            (train_signal, val_signal, "Signal"),
            (train_bkg, val_bkg, "Bkg")
        ]:
            if len(data1) > 10 and len(data2) > 10:
                stat, p = ks_2samp(data1, data2)
                ks_results.append(f"{name} Kolmogorov-Smirnov test: p={p:.3f}")
            else:
                ks_results.append(f"{name} KS: insufficient samples")

        plt.gcf().text(
            0.5, 0.915, '\n'.join(ks_results),
            ha='center', fontsize=9, bbox={'facecolor':'white', 'alpha':0.5}
        )

        plt.tight_layout()
        out_name = f"test_{feature_name}_decomposed.png"
        plt.savefig(out_name, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Generated decomposition plot: {out_name}")
    except Exception as e:
        print(f"Feature {feature_name} decomposition failed: {str(e)}")

def plot_training_metrics(train_losses, val_losses, val_aucs, signal_efficiencies, background_rejections, background_acceptances, background_ratio, out_dir: str = None):
    plt.figure(figsize=(15, 12))
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(val_aucs, label='AUC')
    plt.title('Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(signal_efficiencies, label='Signal Efficiency')
    plt.plot(background_rejections, label='Background Rejection')
    plt.plot(background_ratio, label='Background Ratio')
    plt.title('Efficiency Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(background_acceptances, label='Background Acceptance')
    plt.title('Background Acceptance Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.legend()
    plt.tight_layout()
    try:
        if out_dir:
            outp = Path(out_dir)
            if outp.suffix:
                outp = outp.parent
            outp.mkdir(parents=True, exist_ok=True)
            out_path = outp / "training_metrics.png"
        else:
            out_path = Path("training_metrics.png")
        plt.savefig(str(out_path), dpi=300)
    finally:
        plt.close()

def plot_roc_curve(y_true, y_score, out_path=None, title="ROC Curve", pos_label=1):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Background Acceptance')
    plt.ylabel('Signal Efficiency')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] ROC curve saved to {out_path}")
    else:
        plt.show()

def plot_score_distribution(y_true, y_score, out_path=None, title="Score Distribution", bins=50):
    import matplotlib.pyplot as plt
    import numpy as np

    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    plt.figure(figsize=(8,6))
    plt.hist(pos_scores, bins=bins, alpha=0.6, label=f'Signal', color='blue', density=True)
    plt.hist(neg_scores, bins=bins, alpha=0.6, label=f'Background', color='red', density=True)
    # plt.hist(pos_scores, bins=bins, alpha=0.6, label=f'Signal (N={len(pos_scores)})', color='blue', density=True)
    # plt.hist(neg_scores, bins=bins, alpha=0.6, label=f'Background (N={len(neg_scores)})', color='red', density=True)
    plt.xlabel('Predicted Score (probability of positive class)')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Score distribution saved to {out_path}")
    else:
        plt.show()

def plot_multiclass_score_distribution(y_true, y_score, out_path=None, title="Score Distribution", bins=50, class_names=None):
    """
    Plot score distribution for multiclass classification.
    For each class, plot the distribution of its predicted probability,
    grouped by the true class (all classes shown in the same subplot).
    
    Parameters
    ----------
    y_true : array-like
        True labels (integers)
    y_score : array-like
        Predicted probabilities of shape (n_samples, n_classes)
    out_path : str or None
        Path to save the plot
    title : str
        Plot title
    bins : int
        Number of histogram bins
    class_names : list or None
        List of class names for legend
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true).flatten()
    
    if y_score.ndim == 1:
        # Binary classification case
        plot_score_distribution(y_true, y_score, out_path, title, bins)
        return
    
    n_classes = y_score.shape[1]
    unique_classes = np.unique(y_true)
    # If some classes missing in y_true, still keep all n_classes
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    elif len(class_names) < n_classes:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    # Determine subplot grid
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_classes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Use a color map for true classes
    cmap = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for i in range(n_classes):
        ax = axes[i]
        scores = y_score[:, i]  # predicted probability for class i
        
        # For each true class, plot its score distribution
        for k in range(n_classes):
            mask = (y_true == k)
            if np.sum(mask) == 0:
                continue
            ax.hist(scores[mask], bins=bins, alpha=0.5, 
                    label=f'{class_names[k]} (N={np.sum(mask)})',
                    color=cmap[k], density=True)
        
        ax.set_xlabel(f'Predicted Probability for {class_names[i]}')
        ax.set_ylabel('Density')
        ax.set_title(f'Class {i}: {class_names[i]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_classes, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Multi-class score distribution saved to {out_path}")
    else:
        plt.show()

def plot_multiclass_roc_curve(y_true, y_score, out_path=None, title="Multiclass ROC Curves", class_names=None):
    """
    Plot ROC curves for multiclass classification (one-vs-rest).
    """
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    import numpy as np
    
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true).flatten()
    
    n_classes = y_score.shape[1]
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    elif len(class_names) < n_classes:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for i in range(n_classes):
        y_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'Class {i}: {class_names[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Background Acceptance)')
    plt.ylabel('True Positive Rate (Signal Efficiency)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Multiclass ROC curves saved to {out_path}")
    else:
        plt.show()

def plot_roc_and_distribution(y_true, y_score, out_dir=None, prefix="", pos_label=1, class_names=None):
    """
    Plot ROC curve and score distribution for both binary and multiclass.
    """
    if out_dir is None:
        out_dir = "."
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true).flatten()
    
    if y_score.ndim == 2 and y_score.shape[1] > 1:
        roc_path = os.path.join(out_dir, f"{prefix}roc_curve_multiclass.png")
        dist_path = os.path.join(out_dir, f"{prefix}score_distribution_multiclass.png")
        
        plot_multiclass_roc_curve(y_true, y_score, roc_path, class_names=class_names)
        plot_multiclass_score_distribution(y_true, y_score, dist_path, class_names=class_names)
    else:
        if y_score.ndim == 2 and y_score.shape[1] == 1:
            y_score = y_score.flatten()
        roc_path = os.path.join(out_dir, f"{prefix}roc_curve.png")
        dist_path = os.path.join(out_dir, f"{prefix}score_distribution.png")
        plot_roc_curve(y_true, y_score, roc_path, pos_label=pos_label)
        plot_score_distribution(y_true, y_score, dist_path)