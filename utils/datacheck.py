import numpy as np

def check_data_quality(dataset, name="dataset"):
    """Check data quality of the dataset."""
    print(f"\nChecking data quality for {name}:")
    print(f"Number of samples: {len(dataset)}")
    print(f"Feature dimension: {dataset.X.shape[1] if len(dataset.X.shape) > 1 else 0}")
    
    if len(dataset.X) > 0:
        X_numpy = dataset.X
        nan_count = np.isnan(X_numpy).sum()
        inf_count = np.isinf(X_numpy).sum()
        print(f"NaN count in features: {nan_count}")
        print(f"Infinity count in features: {inf_count}")
        print(f"Feature range: [{X_numpy.min():.6f}, {X_numpy.max():.6f}]")
        print(f"Feature mean: {X_numpy.mean():.6f}")
        print(f"Feature std: {X_numpy.std():.6f}")
    
    if hasattr(dataset, 'y') and len(dataset.y) > 0:
        y_numpy = dataset.y
        nan_count = np.isnan(y_numpy).sum()
        inf_count = np.isinf(y_numpy).sum()
        print(f"NaN count in labels: {nan_count}")
        print(f"Infinity count in labels: {inf_count}")
        print(f"Label range: [{y_numpy.min():.6f}, {y_numpy.max():.6f}]")