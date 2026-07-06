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