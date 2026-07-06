import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_correlation_with_target(feature_names, X_data,
                                    target_names, target_data,
                                    out_dir=None, n_samples=5000,
                                    method='pearson', random_state=42):
    if hasattr(X_data, 'numpy'):
        X = X_data.numpy()
    else:
        X = np.asarray(X_data)
    if hasattr(target_data, 'numpy'):
        Y = target_data.numpy()
    else:
        Y = np.asarray(target_data)
    n_total = X.shape[0]
    if Y.shape[0] != n_total:
        raise ValueError("X_data and target_data must have same number of samples")
    if n_samples is not None and n_total > n_samples:
        rng = np.random.RandomState(random_state)
        indices = rng.choice(n_total, n_samples, replace=False)
        X = X[indices]
        Y = Y[indices]
        print(f"[INFO] Correlation sampled {n_samples} events out of {n_total} total.")
    else:
        print(f"[INFO] Using all {n_total} events for correlation.")
    df_X = pd.DataFrame(X, columns=feature_names)
    df_Y = pd.DataFrame(Y, columns=target_names)
    corr_matrix = df_X.corrwith(df_Y, method=method)
    corr_df = pd.DataFrame(index=feature_names, columns=target_names, dtype=float)
    for f in feature_names:
        for t in target_names:
            corr_df.loc[f, t] = df_X[f].corr(df_Y[t], method=method)
    if out_dir:
        outp = Path(out_dir)
        if outp.suffix:
            outp = outp.parent
        outp.mkdir(parents=True, exist_ok=True)
        csv_path = outp / "feature_target_correlation.csv"
        png_path = outp / "feature_target_correlation_heatmap.png"
    else:
        csv_path = Path("feature_target_correlation.csv")
        png_path = Path("feature_target_correlation_heatmap.png")
    corr_df.to_csv(csv_path)
    print(f"[INFO] Correlation CSV saved to {csv_path}")
    plt.figure(figsize=(max(8, len(target_names)*0.8), max(6, len(feature_names)*0.6)))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                square=True, cbar_kws={"shrink": 0.8})
    plt.xlabel("Target Features")
    plt.ylabel("Model Features")
    plt.title(f"Correlation between Model Features and Target Features ({method})")
    plt.tight_layout()
    plt.savefig(str(png_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Correlation heatmap saved to {png_path}")
    return corr_df