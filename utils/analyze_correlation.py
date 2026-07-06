import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_feature_correlation(feature_names, X_data, out_dir=None, 
                                max_features=None, n_samples=5000, 
                                method='pearson', random_state=42):
    if hasattr(X_data, 'numpy'):
        X = X_data.numpy()
    else:
        X = np.asarray(X_data)
    n_total_samples, n_features = X.shape
    if len(feature_names) != n_features:
        raise ValueError("len(feature_names) != number of features")
    if n_samples is not None and n_total_samples > n_samples:
        rng = np.random.RandomState(random_state)
        indices = rng.choice(n_total_samples, n_samples, replace=False)
        X = X[indices]
        print(f"[INFO] Correlation sampled {n_samples} events out of {n_total_samples} total.")
    else:
        print(f"[INFO] Using all {n_total_samples} events for correlation.")
    if max_features is not None and max_features < n_features:
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-max_features:]
        X = X[:, top_indices]
        feature_names = [feature_names[i] for i in top_indices]
        print(f"[INFO] Reduced to {max_features} features with largest variance.")
    corr_matrix = pd.DataFrame(X, columns=feature_names).corr(method=method)
    if out_dir:
        outp = Path(out_dir)
        if outp.suffix:
            outp = outp.parent
        outp.mkdir(parents=True, exist_ok=True)
        png_path = outp / "feature_correlation_heatmap.png"
        csv_path = outp / "feature_correlation_matrix.csv"
    else:
        png_path = Path("feature_correlation_heatmap.png")
        csv_path = Path("feature_correlation_matrix.csv")
    plt.figure(figsize=(16, 16))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, 
                cbar_kws={"shrink": 0.8})
    plt.title(f"Feature Correlation Matrix ({method})", pad=20)
    plt.tight_layout()
    plt.savefig(str(png_path), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Correlation heatmap saved to {png_path}")
    corr_matrix.to_csv(csv_path)
    print(f"[INFO] Correlation matrix CSV saved to {csv_path}")
    return corr_matrix