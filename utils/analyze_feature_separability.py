import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist


def analyze_feature_separability(feature_names, X_data, y_data,
                                 pos_class=0,
                                 out_dir=None, n_samples=5000,
                                 random_state=42):
    """
    Analyze feature separability between signal and background classes.
    
    Parameters
    ----------
    feature_names : list of str
        Names of input features
    X_data : array-like
        Feature matrix of shape (n_samples, n_features)
    y_data : array-like
        Target labels
    pos_class : int
        Index of the positive/signal class (default: 0)
    out_dir : str or None
        Directory to save output plots and CSV
    n_samples : int or None
        Number of samples to use for analysis (subsampling)
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Dictionary containing separability metrics
    """
    # Convert to numpy arrays
    if hasattr(X_data, 'numpy'):
        X = X_data.numpy()
    else:
        X = np.asarray(X_data)
    
    if hasattr(y_data, 'numpy'):
        y = y_data.numpy()
    else:
        y = np.asarray(y_data)
    
    # Ensure 1D labels
    if y.ndim == 2 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    y = y.flatten()
    
    n_total = X.shape[0]
    
    # Subsampling for large datasets
    if n_samples is not None and n_total > n_samples:
        rng = np.random.RandomState(random_state)
        indices = rng.choice(n_total, n_samples, replace=False)
        X = X[indices]
        y = y[indices]
        print(f"[INFO] Separability sampled {n_samples} events out of {n_total} total.")
    else:
        print(f"[INFO] Using all {n_total} events for separability analysis.")
    
    # Separate signal and background using pos_class
    sig_mask = (y == pos_class)
    bg_mask = (y != pos_class)
    n_sig = sig_mask.sum()
    n_bg = bg_mask.sum()
    
    if n_sig == 0 or n_bg == 0:
        print(f"[WARN] Signal (class {pos_class}) or background samples count is 0, skipping analysis.")
        return None
    
    X_sig = X[sig_mask]
    X_bg = X[bg_mask]
    
    # ========== 1. Within-class distances ==========
    sig_within_dist = np.mean(pdist(X_sig)) if n_sig > 1 else 0.0
    bg_within_dist = np.mean(pdist(X_bg)) if n_bg > 1 else 0.0
    avg_within = (sig_within_dist + bg_within_dist) / 2
    
    # ========== 2. Between-class distance ==========
    sig_center = X_sig.mean(axis=0)
    bg_center = X_bg.mean(axis=0)
    between_dist = np.linalg.norm(sig_center - bg_center)
    
    # ========== 3. Separability index ==========
    separability = between_dist / (avg_within + 1e-8)
    
    # ========== 4. Per-feature Cohen's d ==========
    cohens_d = []
    for i in range(X.shape[1]):
        mu_sig = X_sig[:, i].mean()
        mu_bg = X_bg[:, i].mean()
        std_sig = X_sig[:, i].std()
        std_bg = X_bg[:, i].std()
        pooled_std = np.sqrt((std_sig**2 + std_bg**2) / 2)
        d = abs(mu_sig - mu_bg) / (pooled_std + 1e-8)
        cohens_d.append(d)
    
    # ========== 5. Per-feature AUC ==========
    feature_auc = []
    for i in range(X.shape[1]):
        try:
            auc = roc_auc_score(y, X[:, i])
            if auc < 0.5:
                auc = 1.0 - auc
            feature_auc.append(auc)
        except Exception:
            feature_auc.append(0.5)
    
    # ========== 6. Build results dataframe ==========
    results_df = pd.DataFrame({
        'feature': feature_names,
        'cohens_d': cohens_d,
        'auc': feature_auc
    }).sort_values('auc', ascending=False)
    
    # ========== 7. Print summary ==========
    print(f"\n{'='*70}")
    print("FEATURE SEPARABILITY ANALYSIS")
    print(f"{'='*70}")
    print(f"Positive class (signal): {pos_class}")
    print(f"Total samples: {n_total} (signal: {n_sig}, background: {n_bg})")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"\n--- Distance Metrics ---")
    print(f"Signal within-class distance:  {sig_within_dist:.4f}")
    print(f"Background within-class dist:  {bg_within_dist:.4f}")
    print(f"Between-class distance:        {between_dist:.4f}")
    print(f"Separability index:            {separability:.4f}")
    print(f"(Guide: <0.5=poor, 0.5-1.0=moderate, >1.0=good)")
    
    top5 = results_df.head(5)
    print(f"\n--- Top 5 Most Discriminative Features ---")
    print(top5[['feature', 'auc', 'cohens_d']].to_string(index=False))
    
    n_gt_06 = (np.array(feature_auc) > 0.6).sum()
    n_gt_07 = (np.array(feature_auc) > 0.7).sum()
    print(f"\n--- Feature AUC Summary ---")
    print(f"Mean AUC: {np.mean(feature_auc):.4f}")
    print(f"Median AUC: {np.median(feature_auc):.4f}")
    print(f"AUC > 0.6: {n_gt_06}/{len(feature_auc)} features")
    print(f"AUC > 0.7: {n_gt_07}/{len(feature_auc)} features")
    print(f"{'='*70}\n")
    
    # ========== 8. Save outputs ==========
    if out_dir:
        outp = Path(out_dir)
        if outp.suffix:
            outp = outp.parent
        outp.mkdir(parents=True, exist_ok=True)
        csv_path = outp / "feature_separability.csv"
        png_pca_path = outp / "feature_separability_pca.png"
        png_bar_path = outp / "feature_separability_barchart.png"
        png_heatmap_path = outp / "feature_separability_heatmap.png"
    else:
        csv_path = Path("feature_separability.csv")
        png_pca_path = Path("feature_separability_pca.png")
        png_bar_path = Path("feature_separability_barchart.png")
        png_heatmap_path = Path("feature_separability_heatmap.png")
    
    # Save CSV
    results_df.to_csv(csv_path, index=False)
    print(f"[INFO] Separability CSV saved to {csv_path}")
    
    # ========== 9. Bar chart of top features ==========
    top_n = min(30, len(results_df))
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    top_df = results_df.head(top_n).sort_values('auc', ascending=True)
    bars = ax.barh(top_df['feature'], top_df['auc'], color='steelblue')
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1, label='Random (AUC=0.5)')
    ax.axvline(x=0.7, color='green', linestyle='--', linewidth=1, label='Good (AUC=0.7)')
    ax.set_xlabel('AUC (Single Feature)')
    ax.set_title(f'Top {top_n} Features by Discriminative Power')
    ax.legend(loc='lower right')
    ax.set_xlim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(str(png_bar_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Feature barchart saved to {png_bar_path}")
    
    # ========== 10. Correlation heatmap of top features ==========
    try:
        top_heatmap_n = min(20, len(results_df))
        top_features = results_df.head(top_heatmap_n)['feature'].tolist()
        top_indices = [feature_names.index(f) for f in top_features]
        X_top = X[:, top_indices]
        
        corr_matrix = np.corrcoef(X_top.T)
        corr_df = pd.DataFrame(corr_matrix, index=top_features, columns=top_features)
        
        fig, ax = plt.subplots(figsize=(max(10, top_heatmap_n * 0.5), 
                                        max(8, top_heatmap_n * 0.5)))
        sns.heatmap(corr_df, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                    square=True, cbar_kws={"shrink": 0.8},
                    annot_kws={'size': 8})
        plt.title(f'Correlation Matrix of Top {top_heatmap_n} Features')
        plt.tight_layout()
        plt.savefig(str(png_heatmap_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Feature correlation heatmap saved to {png_heatmap_path}")
    except Exception as e:
        print(f"[WARN] Correlation heatmap failed: {e}")
    
    # ========== 11. PCA Visualization ==========
    try:
        n_components = min(2, X.shape[1])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(X_pca[sig_mask, 0], X_pca[sig_mask, 1],
                   c='red', label='Signal', alpha=0.4, s=3)
        ax.scatter(X_pca[bg_mask, 0], X_pca[bg_mask, 1],
                   c='blue', label='Background', alpha=0.4, s=3)
        
        # Plot class centroids
        sig_pca_center = X_pca[sig_mask].mean(axis=0)
        bg_pca_center = X_pca[bg_mask].mean(axis=0)
        ax.scatter(*sig_pca_center, c='red', marker='X', s=200, 
                   edgecolors='black', linewidth=2, label='Signal centroid')
        ax.scatter(*bg_pca_center, c='blue', marker='X', s=200, 
                   edgecolors='black', linewidth=2, label='Background centroid')
        
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)' if n_components > 1 else 'PC2')
        ax.set_title('Feature Separability - PCA Projection')
        ax.legend(loc='best')
        
        # Add separability info on plot
        textstr = f'Separability: {separability:.3f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(str(png_pca_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] PCA visualization saved to {png_pca_path}")
        
    except Exception as e:
        print(f"[WARN] PCA visualization failed: {e}")
    
    return {
        'separability': separability,
        'between_dist': between_dist,
        'within_dist_signal': sig_within_dist,
        'within_dist_background': bg_within_dist,
        'best_feature': results_df.iloc[0]['feature'],
        'best_auc': results_df.iloc[0]['auc'],
        'feature_aucs': feature_auc,
        'cohens_d': cohens_d,
        'results_df': results_df
    }