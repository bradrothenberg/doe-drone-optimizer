"""
Feature Engineering for DOE Drone Design Dataset
Creates derived features to improve model performance
"""

import numpy as np
import pandas as pd
from typing import Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Engineer features from raw geometric design parameters

    Raw features (7):
    - LOA (Length Overall)
    - Span
    - LE Sweep P1/P2 (Leading Edge sweeps)
    - TE Sweep P1/P2 (Trailing Edge sweeps)
    - Panel Break (span fraction)

    Derived features (10):
    - Aspect ratio proxy
    - Sweep differentials (taper indicators)
    - Average sweeps
    - Sweep asymmetry
    - Wing loading proxy
    - Planform complexity
    - Span/LOA ratio
    - Panel break interactions
    """

    def __init__(self):
        self.feature_names = []

    def fit(self, X: np.ndarray, feature_names: list = None) -> 'FeatureEngineer':
        """
        Fit feature engineer (compute feature names)

        Args:
            X: Raw input features (n_samples, 7)
            feature_names: List of raw feature names

        Returns:
            self
        """
        if feature_names is None:
            feature_names = [
                'LOA', 'Span', 'LE_Sweep_P1', 'LE_Sweep_P2',
                'TE_Sweep_P1', 'TE_Sweep_P2', 'Panel_Break'
            ]

        self.raw_feature_names = feature_names
        self.feature_names = self._get_all_feature_names()

        logger.info(f"Feature engineering: {len(self.raw_feature_names)} raw → "
                   f"{len(self.feature_names)} engineered features")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform raw features into engineered features

        Args:
            X: Raw input features (n_samples, 7)
               Columns: [LOA, Span, LE_Sweep_P1, LE_Sweep_P2,
                        TE_Sweep_P1, TE_Sweep_P2, Panel_Break]

        Returns:
            Engineered features (n_samples, 17)
        """
        # Extract raw features
        loa = X[:, 0]          # Length Overall (inches)
        span = X[:, 1]         # Wing span (inches)
        le_sweep_p1 = X[:, 2]  # LE sweep panel 1 (degrees)
        le_sweep_p2 = X[:, 3]  # LE sweep panel 2 (degrees)
        te_sweep_p1 = X[:, 4]  # TE sweep panel 1 (degrees)
        te_sweep_p2 = X[:, 5]  # TE sweep panel 2 (degrees)
        panel_break = X[:, 6]  # Panel break location (fraction 0-1)

        # Derived features list
        features = []

        # 1. Raw features (7)
        features.extend([
            loa, span, le_sweep_p1, le_sweep_p2,
            te_sweep_p1, te_sweep_p2, panel_break
        ])

        # 2. Aspect ratio proxy (span^2 / wing_area_estimate)
        # Approximate wing area as loa * span (rectangular approximation)
        wing_area_proxy = loa * span
        aspect_ratio_proxy = span ** 2 / (wing_area_proxy + 1e-6)
        features.append(aspect_ratio_proxy)

        # 3. Sweep differentials (indicate wing taper)
        sweep_diff_p1 = te_sweep_p1 - le_sweep_p1
        sweep_diff_p2 = te_sweep_p2 - le_sweep_p2
        features.extend([sweep_diff_p1, sweep_diff_p2])

        # 4. Average sweeps
        avg_le_sweep = (le_sweep_p1 + le_sweep_p2) / 2.0
        avg_te_sweep = (te_sweep_p1 + te_sweep_p2) / 2.0
        features.extend([avg_le_sweep, avg_te_sweep])

        # 5. Sweep asymmetry (difference between panels)
        sweep_asymmetry = np.abs(le_sweep_p1 - le_sweep_p2)
        features.append(sweep_asymmetry)

        # 6. Wing loading proxy (inverse of wing area)
        wing_loading_proxy = 1000.0 / (wing_area_proxy + 1e-6)
        features.append(wing_loading_proxy)

        # 7. Planform complexity (sum of absolute sweep differentials)
        planform_complexity = np.abs(sweep_diff_p1) + np.abs(sweep_diff_p2)
        features.append(planform_complexity)

        # 8. Span to LOA ratio (wing slenderness)
        span_loa_ratio = span / (loa + 1e-6)
        features.append(span_loa_ratio)

        # 9. Panel break interaction with sweep
        panel_break_sweep_interaction = panel_break * avg_le_sweep
        features.append(panel_break_sweep_interaction)

        # Stack all features
        X_engineered = np.column_stack(features)

        return X_engineered

    def fit_transform(self, X: np.ndarray, feature_names: list = None) -> np.ndarray:
        """
        Fit and transform in one step

        Args:
            X: Raw input features (n_samples, 7)
            feature_names: List of raw feature names

        Returns:
            Engineered features (n_samples, 17)
        """
        self.fit(X, feature_names)
        return self.transform(X)

    def get_feature_names(self) -> list:
        """Get list of all engineered feature names"""
        return self.feature_names.copy()

    def _get_all_feature_names(self) -> list:
        """Generate list of all feature names"""
        names = self.raw_feature_names.copy()

        # Add derived feature names
        names.extend([
            'Aspect_Ratio_Proxy',
            'Sweep_Diff_P1',
            'Sweep_Diff_P2',
            'Avg_LE_Sweep',
            'Avg_TE_Sweep',
            'Sweep_Asymmetry',
            'Wing_Loading_Proxy',
            'Planform_Complexity',
            'Span_LOA_Ratio',
            'Panel_Break_Sweep_Interaction'
        ])

        return names


def engineer_features(
    X: np.ndarray,
    feature_names: list = None,
    return_engineer: bool = False
) -> Union[np.ndarray, tuple]:
    """
    Convenience function to engineer features

    Args:
        X: Raw input features (n_samples, 7)
        feature_names: List of raw feature names
        return_engineer: If True, return (X_engineered, engineer) tuple

    Returns:
        X_engineered: Engineered features (n_samples, 17)
        or
        (X_engineered, engineer): Tuple if return_engineer=True
    """
    engineer = FeatureEngineer()
    X_engineered = engineer.fit_transform(X, feature_names)

    if return_engineer:
        return X_engineered, engineer
    else:
        return X_engineered


# Feature importance analysis utilities
def analyze_feature_correlations(X: np.ndarray, y: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Analyze correlations between features and targets

    Args:
        X: Engineered features (n_samples, n_features)
        y: Target outputs (n_samples, n_outputs)
        feature_names: List of feature names

    Returns:
        DataFrame with correlation coefficients
    """
    from scipy.stats import pearsonr

    n_features = X.shape[1]
    n_outputs = y.shape[1]

    correlations = np.zeros((n_features, n_outputs))
    p_values = np.zeros((n_features, n_outputs))

    for i in range(n_features):
        for j in range(n_outputs):
            corr, p_val = pearsonr(X[:, i], y[:, j])
            correlations[i, j] = corr
            p_values[i, j] = p_val

    # Create DataFrame
    output_names = ['Range', 'Endurance', 'MTOW', 'Cost']
    df_corr = pd.DataFrame(
        correlations,
        index=feature_names,
        columns=output_names
    )

    return df_corr


if __name__ == "__main__":
    # Test feature engineering
    from app.models.data_loader import load_doe_data

    print("Loading DOE data...")
    loader, X_train, X_val, X_test, y_train, y_val, y_test = load_doe_data()

    # Get raw features (unscaled for feature engineering)
    X_train_raw = loader.X_train
    X_val_raw = loader.X_val
    X_test_raw = loader.X_test

    print("\nEngineering features...")
    engineer = FeatureEngineer()
    X_train_eng = engineer.fit_transform(X_train_raw, loader.get_feature_names())
    X_val_eng = engineer.transform(X_val_raw)
    X_test_eng = engineer.transform(X_test_raw)

    print(f"\nFeature engineering results:")
    print(f"Raw features: {X_train_raw.shape}")
    print(f"Engineered features: {X_train_eng.shape}")
    print(f"\nFeature names ({len(engineer.get_feature_names())}):")
    for i, name in enumerate(engineer.get_feature_names()):
        print(f"  {i+1}. {name}")

    # Analyze correlations
    print("\nAnalyzing feature correlations with targets...")
    df_corr = analyze_feature_correlations(
        X_train_eng,
        loader.y_train,
        engineer.get_feature_names()
    )

    print("\nTop 5 features correlated with each target:")
    for col in df_corr.columns:
        print(f"\n{col}:")
        top_features = df_corr[col].abs().sort_values(ascending=False).head(5)
        for feat, corr in top_features.items():
            print(f"  {feat}: {corr:.3f}")
