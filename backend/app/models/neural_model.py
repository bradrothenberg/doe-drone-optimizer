"""
Neural Network (MLP) Multi-Output Regressor for DOE Drone Design Optimization
Complements XGBoost by capturing non-linear interactions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DroneMLPModel(nn.Module):
    """
    Multi-Layer Perceptron for drone performance prediction

    Architecture:
    Input (17) → Dense(128) → ReLU → BatchNorm → Dropout(0.2)
              → Dense(64)  → ReLU → BatchNorm → Dropout(0.2)
              → Dense(32)  → ReLU → BatchNorm
              → Dense(4)   → Output
    """

    def __init__(
        self,
        input_dim: int = 17,
        hidden_dims: list = [128, 64, 32],
        output_dim: int = 4,
        dropout_rate: float = 0.2
    ):
        """
        Initialize neural network

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer sizes
            output_dim: Number of output targets
            dropout_rate: Dropout probability
        """
        super(DroneMLPModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Dense layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))

            # Dropout (except last hidden layer)
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Output layer (linear, no activation)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass"""
        return self.model(x)


class NeuralNetworkDroneModel:
    """
    Wrapper for PyTorch neural network model
    """

    def __init__(
        self,
        input_dim: int = 17,
        hidden_dims: list = [128, 64, 32],
        output_dim: int = 4,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 200,
        early_stopping_patience: int = 20,
        device: str = None
    ):
        """
        Initialize neural network model

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer sizes
            output_dim: Number of output targets
            dropout_rate: Dropout probability
            learning_rate: Learning rate for Adam optimizer
            weight_decay: L2 regularization strength
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        # Device selection
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Model
        self.model = DroneMLPModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout_rate=dropout_rate
        ).to(self.device)

        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

        # Optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        self.is_fitted = False
        self.training_history = {'train_loss': [], 'val_loss': []}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> 'NeuralNetworkDroneModel':
        """
        Train neural network

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples, 4)
            X_val: Validation features (optional, for early stopping)
            y_val: Validation targets (optional, for early stopping)

        Returns:
            self
        """
        logger.info("Training Neural Network...")
        logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)

        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Validation data
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            has_validation = True
            logger.info(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
        else:
            has_validation = False
            logger.warning("No validation set provided, early stopping disabled")

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_losses = []

            for X_batch, y_batch in train_loader:
                # Forward pass
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

            # Average training loss
            avg_train_loss = np.mean(train_losses)
            self.training_history['train_loss'].append(avg_train_loss)

            # Validation phase
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    y_val_pred = self.model(X_val_tensor)
                    val_loss = self.criterion(y_val_pred, y_val_tensor).item()

                self.training_history['val_loss'].append(val_loss)

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1

                # Log progress
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs}: "
                              f"Train Loss = {avg_train_loss:.6f}, "
                              f"Val Loss = {val_loss:.6f}")

                # Early stopping
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    # Restore best model
                    self.model.load_state_dict(self.best_model_state)
                    break
            else:
                # No validation, just log training loss
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs}: "
                              f"Train Loss = {avg_train_loss:.6f}")

        self.is_fitted = True
        logger.info("Neural Network training complete")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict drone performance metrics

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Predictions (n_samples, 4): [Range, Endurance, MTOW, Cost]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_pred_tensor = self.model(X_tensor)
            y_pred = y_pred_tensor.cpu().numpy()

        return y_pred

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        output_names: list = None
    ) -> Dict[str, Any]:
        """
        Evaluate model performance

        Args:
            X: Input features
            y_true: True target values
            output_names: Names of output variables

        Returns:
            Dictionary with evaluation metrics
        """
        if output_names is None:
            output_names = ['Range', 'Endurance', 'MTOW', 'Cost']

        y_pred = self.predict(X)

        metrics = {
            'overall': {},
            'per_output': {}
        }

        # Overall metrics
        metrics['overall']['r2'] = float(r2_score(y_true, y_pred))
        metrics['overall']['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['overall']['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        # Per-output metrics
        for i, name in enumerate(output_names):
            metrics['per_output'][name] = {
                'r2': float(r2_score(y_true[:, i], y_pred[:, i])),
                'mae': float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
                'rmse': float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
                'mape': float(np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / (y_true[:, i] + 1e-6))) * 100)
            }

        return metrics

    def save(self, output_path: Path) -> None:
        """
        Save model to disk

        Args:
            output_path: Path to save model file (.pt)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'model_config': {
                'input_dim': self.model.input_dim,
                'output_dim': self.model.output_dim
            }
        }, output_path)

        logger.info(f"Saved Neural Network model to {output_path}")

        # Save metadata
        metadata = {
            'model_type': 'Neural Network (MLP)',
            'n_outputs': 4,
            'output_names': ['Range', 'Endurance', 'MTOW', 'Cost'],
            'is_fitted': self.is_fitted,
            'architecture': str(self.model)
        }

        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load(self, input_path: Path) -> None:
        """
        Load model from disk

        Args:
            input_path: Path to model file (.pt)
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Model file not found: {input_path}")

        checkpoint = torch.load(input_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.is_fitted = True

        logger.info(f"Loaded Neural Network model from {input_path}")


def train_neural_network_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    **nn_params
) -> Tuple[NeuralNetworkDroneModel, Dict[str, Any]]:
    """
    Convenience function to train and evaluate neural network

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        **nn_params: Neural network hyperparameters

    Returns:
        Tuple of (trained_model, validation_metrics)
    """
    model = NeuralNetworkDroneModel(**nn_params)
    model.fit(X_train, y_train, X_val, y_val)

    # Evaluate on validation set
    if X_val is not None and y_val is not None:
        metrics = model.evaluate(X_val, y_val)
        logger.info(f"\nValidation Metrics:")
        logger.info(f"Overall R²: {metrics['overall']['r2']:.4f}")
        logger.info(f"Overall MAE: {metrics['overall']['mae']:.4f}")
        logger.info(f"\nPer-output R² scores:")
        for name, output_metrics in metrics['per_output'].items():
            logger.info(f"  {name}: {output_metrics['r2']:.4f}")
    else:
        metrics = None

    return model, metrics


if __name__ == "__main__":
    # Test Neural Network model
    from app.models.data_loader import load_doe_data
    from app.models.feature_engineering import engineer_features

    print("Loading DOE data...")
    loader, _, _, _, _, _, _ = load_doe_data()

    # Engineer features
    print("\nEngineering features...")
    X_train_eng, engineer = engineer_features(loader.X_train, loader.get_feature_names(), return_engineer=True)
    X_val_eng = engineer.transform(loader.X_val)
    X_test_eng = engineer.transform(loader.X_test)

    # Scale features for neural network
    X_train_scaled, y_train_scaled = loader.transform(X_train_eng, loader.y_train)
    X_val_scaled, y_val_scaled = loader.transform(X_val_eng, loader.y_val)
    X_test_scaled, y_test_scaled = loader.transform(X_test_eng, loader.y_test)

    # Train Neural Network
    print("\nTraining Neural Network...")
    model, val_metrics = train_neural_network_model(
        X_train_scaled, y_train_scaled,
        X_val_scaled, y_val_scaled,
        input_dim=X_train_scaled.shape[1],
        epochs=100  # Reduced for testing
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = model.evaluate(X_test_scaled, y_test_scaled, loader.get_output_names())

    print(f"\nTest Set Results:")
    print(f"Overall R²: {test_metrics['overall']['r2']:.4f}")
    print(f"Overall MAE: {test_metrics['overall']['mae']:.4f}")
    print(f"\nPer-output metrics:")
    for name, metrics in test_metrics['per_output'].items():
        print(f"\n{name}:")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
