import numpy as np
from scipy import stats
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif, f_classif


def analyze_target(y: np.ndarray, random_state: int = 42) -> dict:
    np.random.seed(random_state)
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()

    return {
        "n_classes": int(len(values)),
        "counts": dict(zip(values.tolist(), counts.tolist())),
        "percentages": dict(zip(values.tolist(), (probs * 100).tolist())),
        "imbalance_ratio": float(counts.max() / counts.min()),
        "entropy": float(entropy(probs)),
        "gini": float(1.0 - np.sum(probs**2)),
    }


def analyze_missing(X: np.ndarray, random_state: int = 42) -> dict:
    np.random.seed(random_state)
    nan_mask = np.isnan(X)

    return {
        "total_missing": int(nan_mask.sum()),
        "missing_ratio": float(nan_mask.mean()),
        "per_feature_ratio": nan_mask.mean(axis=0).tolist(),
    }


def analyze_variance(X: np.ndarray, random_state: int = 42) -> dict:
    np.random.seed(random_state)
    var = np.nanvar(X, axis=0)

    return {
        "mean_variance": float(var.mean()),
        "low_variance_ratio": float((var < 1e-6).mean()),
        "min_variance": float(var.min()),
        "max_variance": float(var.max()),
    }


def analyze_correlation(X: np.ndarray, random_state: int = 42) -> dict:
    np.random.seed(random_state)
    corr = np.corrcoef(X, rowvar=False)
    upper = np.abs(corr[np.triu_indices_from(corr, k=1)])

    return {
        "mean_abs_corr": float(np.nanmean(upper)),
        "high_corr_ratio": float((upper > 0.9).mean()),
        "max_corr": float(np.nanmax(upper)),
    }


def analyze_informativeness(
    X: np.ndarray, y: np.ndarray, random_state: int = 42
) -> dict:
    np.random.seed(random_state)
    mi = mutual_info_classif(X, y, discrete_features=False)

    return {
        "mean_mi": float(mi.mean()),
        "zero_mi_ratio": float((mi < 1e-4).mean()),
        "max_mi": float(mi.max()),
    }


def analyze_outliers(
    X: np.ndarray, method: str = "iqr", threshold: float = 1.5, random_state: int = 42
) -> dict:
    np.random.seed(random_state)
    outlier_stats = {}

    if method == "iqr":
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers_per_feature = np.sum((X < lower_bound) | (X > upper_bound), axis=0)
        outlier_ratio = outliers_per_feature / X.shape[0]

    elif method == "zscore":
        z_scores = np.abs(stats.zscore(X, nan_policy="omit"))
        outliers_per_feature = np.sum(z_scores > threshold, axis=0)
        outlier_ratio = outliers_per_feature / X.shape[0]

    outlier_stats["outlier_ratio_per_feature"] = outlier_ratio.tolist()
    outlier_stats["mean_outlier_ratio"] = np.mean(outlier_ratio)
    outlier_stats["features_with_high_outliers"] = np.sum(outlier_ratio > 0.05)

    return outlier_stats


def analyze_feature_types(X: np.ndarray, random_state: int = 42) -> dict:
    np.random.seed(random_state)
    n_features = X.shape[1]
    stats = {}

    binary_mask = np.array([len(np.unique(X[:, i])) == 2 for i in range(n_features)])
    stats["n_binary_features"] = np.sum(binary_mask)
    stats["binary_ratio"] = np.sum(binary_mask) / n_features

    categorical_mask = np.array(
        [len(np.unique(X[:, i])) <= 10 for i in range(n_features)]
    )
    stats["n_categorical_features"] = np.sum(categorical_mask)
    stats["categorical_ratio"] = np.sum(categorical_mask) / n_features

    continuous_mask = ~categorical_mask
    stats["n_continuous_features"] = np.sum(continuous_mask)
    stats["continuous_ratio"] = np.sum(continuous_mask) / n_features

    return stats


def analyze_feature_target_relationship(
    X: np.ndarray, y: np.ndarray, random_state: int = 42
) -> dict:
    np.random.seed(random_state)
    f_scores, p_values = f_classif(X, y)
    mi_scores = mutual_info_classif(X, y, random_state=42)

    stats = {
        "f_scores": f_scores.tolist(),
        "p_values": p_values.tolist(),
        "mi_scores": mi_scores.tolist(),
        "n_significant_features": np.sum(p_values < 0.05),
        "top_f_features": np.argsort(f_scores)[-5:][::-1].tolist(),
        "top_mi_features": np.argsort(mi_scores)[-5:][::-1].tolist(),
    }

    return stats
