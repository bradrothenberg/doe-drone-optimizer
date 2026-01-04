"""
Data Loader and Preprocessing for DOE Drone Design Dataset
Loads doe_summary.csv and prepares data for ML training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DOEDataLoader:
    """Load and preprocess DOE dataset for ML training"""

    # Define column mappings from doe_summary.csv
    INPUT_FEATURES = [
        'loa',              # Length Overall (inches)
        'span',             # Wing Span (inches)
        'le_sweep_p1',      # Leading Edge Sweep Panel 1 (degrees)
        'le_sweep_p2',      # Leading Edge Sweep Panel 2 (degrees)
        'te_sweep_p1',      # Trailing Edge Sweep Panel 1 (degrees)
        'te_sweep_p2',      # Trailing Edge Sweep Panel 2 (degrees)
        'panel_break'       # Panel break location (fraction)
    ]

    PRIMARY_OUTPUTS = [
        'range_nm',                  # Range in nautical miles
        'endurance_hr',              # Endurance in hours
        'mtow_lbm',                  # Max Takeoff Weight in pounds
        'material_cost_usd',         # Material cost in USD
        'wingtip_deflection_in'      # Wingtip deflection in inches
    ]

    ADDITIONAL_OUTPUTS = [
        'ld_cruise',
        'ld_max',
        'aspect_ratio',
        'fuel_fraction',
        'static_margin_pct',
        'wingtip_deflection_in'
    ]

    def __init__(self, data_path: str = None):
        """
        Initialize data loader

        Args:
            data_path: Path to doe_summary.csv (default: backend/data/doe_summary.csv)
        """
        if data_path is None:
            # Default path relative to this file
            base_dir = Path(__file__).parent.parent.parent
            data_path = base_dir / "data" / "doe_summary.csv"

        self.data_path = Path(data_path)
        self.scaler_X = RobustScaler()  # Robust to outliers
        self.scaler_y = RobustScaler()

        self.df_raw = None
        self.df_clean = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def load_data(self) -> pd.DataFrame:
        """
        Load raw data from CSV

        Returns:
            DataFrame with all DOE samples
        """
        logger.info(f"Loading data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.df_raw = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df_raw)} samples with {len(self.df_raw.columns)} columns")

        return self.df_raw

    def clean_data(self) -> pd.DataFrame:
        """
        Clean data by removing infeasible designs

        Returns:
            Cleaned DataFrame
        """
        if self.df_raw is None:
            self.load_data()

        logger.info("Cleaning data...")

        # Remove designs with zero or negative range (infeasible)
        mask_feasible = self.df_raw['range_nm'] > 0
        n_removed = (~mask_feasible).sum()

        self.df_clean = self.df_raw[mask_feasible].copy()

        logger.info(f"Removed {n_removed} infeasible designs (range <= 0)")
        logger.info(f"Remaining samples: {len(self.df_clean)}")

        # Check for missing values
        missing = self.df_clean[self.INPUT_FEATURES + self.PRIMARY_OUTPUTS].isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"Found missing values:\n{missing[missing > 0]}")
            self.df_clean = self.df_clean.dropna(subset=self.INPUT_FEATURES + self.PRIMARY_OUTPUTS)
            logger.info(f"After removing NaN: {len(self.df_clean)} samples")

        return self.df_clean

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics

        Returns:
            Dictionary with statistics for inputs and outputs
        """
        if self.df_clean is None:
            self.clean_data()

        stats = {
            'n_samples': len(self.df_clean),
            'inputs': {},
            'outputs': {}
        }

        # Input statistics
        for feature in self.INPUT_FEATURES:
            stats['inputs'][feature] = {
                'min': float(self.df_clean[feature].min()),
                'max': float(self.df_clean[feature].max()),
                'mean': float(self.df_clean[feature].mean()),
                'std': float(self.df_clean[feature].std())
            }

        # Output statistics
        for output in self.PRIMARY_OUTPUTS:
            stats['outputs'][output] = {
                'min': float(self.df_clean[output].min()),
                'max': float(self.df_clean[output].max()),
                'mean': float(self.df_clean[output].mean()),
                'std': float(self.df_clean[output].std())
            }

        return stats

    def prepare_train_test_split(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/validation/test sets

        Args:
            test_size: Fraction for test set (default: 0.15)
            val_size: Fraction for validation set (default: 0.15)
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if self.df_clean is None:
            self.clean_data()

        logger.info("Preparing train/validation/test split...")

        # Extract features and targets
        X = self.df_clean[self.INPUT_FEATURES].values
        y = self.df_clean[self.PRIMARY_OUTPUTS].values

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Second split: separate validation set from remaining data
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for already removed test set
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )

        logger.info(f"Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        logger.info(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        logger.info(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        return X_train, X_val, X_test, y_train, y_val, y_test

    def fit_scalers(self) -> None:
        """Fit scalers on training data"""
        if self.X_train is None:
            self.prepare_train_test_split()

        logger.info("Fitting scalers on training data...")
        self.scaler_X.fit(self.X_train)
        self.scaler_y.fit(self.y_train)

    def transform(
        self,
        X: np.ndarray = None,
        y: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform features and targets using fitted scalers

        Args:
            X: Input features to transform (if None, uses stored data)
            y: Output targets to transform (if None, uses stored data)

        Returns:
            Tuple of (X_scaled, y_scaled)
        """
        X_scaled = self.scaler_X.transform(X) if X is not None else None
        y_scaled = self.scaler_y.transform(y) if y is not None else None

        return X_scaled, y_scaled

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform predictions back to original scale

        Args:
            y_scaled: Scaled predictions

        Returns:
            Predictions in original scale
        """
        return self.scaler_y.inverse_transform(y_scaled)

    def get_feature_names(self) -> list:
        """Get list of input feature names"""
        return self.INPUT_FEATURES.copy()

    def get_output_names(self) -> list:
        """Get list of output target names"""
        return self.PRIMARY_OUTPUTS.copy()

    def save_scalers(self, output_dir: Path) -> None:
        """
        Save fitted scalers to disk

        Args:
            output_dir: Directory to save scaler files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scaler_X_path = output_dir / "scaler_X.pkl"
        scaler_y_path = output_dir / "scaler_y.pkl"

        joblib.dump(self.scaler_X, scaler_X_path)
        joblib.dump(self.scaler_y, scaler_y_path)

        logger.info(f"Saved scalers to {output_dir}")

    def load_scalers(self, input_dir: Path) -> None:
        """
        Load fitted scalers from disk

        Args:
            input_dir: Directory containing scaler files
        """
        input_dir = Path(input_dir)

        scaler_X_path = input_dir / "scaler_X.pkl"
        scaler_y_path = input_dir / "scaler_y.pkl"

        if not scaler_X_path.exists() or not scaler_y_path.exists():
            raise FileNotFoundError(f"Scaler files not found in {input_dir}")

        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)

        logger.info(f"Loaded scalers from {input_dir}")


# Convenience function for quick data loading
def load_doe_data(
    data_path: str = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[DOEDataLoader, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to load and prepare DOE data in one call

    Args:
        data_path: Path to doe_summary.csv
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed

    Returns:
        Tuple of (data_loader, X_train, X_val, X_test, y_train, y_val, y_test)
    """
    loader = DOEDataLoader(data_path)
    loader.load_data()
    loader.clean_data()

    # Log statistics
    stats = loader.get_statistics()
    logger.info(f"\nDataset Statistics:")
    logger.info(f"Total samples: {stats['n_samples']}")
    logger.info(f"\nInput ranges:")
    for feature, stat in stats['inputs'].items():
        logger.info(f"  {feature}: [{stat['min']:.2f}, {stat['max']:.2f}]")
    logger.info(f"\nOutput ranges:")
    for output, stat in stats['outputs'].items():
        logger.info(f"  {output}: [{stat['min']:.2f}, {stat['max']:.2f}]")

    # Split and scale
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_train_test_split(
        test_size=test_size, val_size=val_size, random_state=random_state
    )

    loader.fit_scalers()

    # Transform data
    X_train_scaled, y_train_scaled = loader.transform(X_train, y_train)
    X_val_scaled, y_val_scaled = loader.transform(X_val, y_val)
    X_test_scaled, y_test_scaled = loader.transform(X_test, y_test)

    return loader, X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled


if __name__ == "__main__":
    # Test data loader
    loader, X_train, X_val, X_test, y_train, y_val, y_test = load_doe_data()

    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"y_test: {y_test.shape}")

    print(f"\nFeature names: {loader.get_feature_names()}")
    print(f"Output names: {loader.get_output_names()}")
