#!/usr/bin/env python3
"""
Model training module for IoT Network Threat Detection.
Implements Random Forest and XGBoost classifiers for threat detection.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IoTThreatDetector:
    """ML model for IoT network threat detection."""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load preprocessed data and split features/labels."""
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Assume the target column is 'label_encoded' or 'label'
        target_col = 'label_encoded' if 'label_encoded' in df.columns else 'label'
        
        if target_col not in df.columns:
            raise ValueError(f"No target column found. Expected 'label' or 'label_encoded'")
        
        # Separate features and target
        y = df[target_col]
        X = df.drop([target_col, 'label'], axis=1, errors='ignore')
        
        self.feature_columns = X.columns.tolist()
        logger.info(f"Loaded {len(X)} samples with {len(X.columns)} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def preprocess_features(self, X: pd.DataFrame, fit_scaler: bool = True) -> np.ndarray:
        """Scale features for model training."""
        logger.info("Preprocessing features...")
        
        # Handle categorical features if any
        X_processed = X.copy()
        
        # Convert categorical columns to numeric
        for col in X_processed.select_dtypes(include=['object']).columns:
            X_processed[col] = pd.Categorical(X_processed[col]).codes
        
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_processed)
        else:
            X_scaled = self.scaler.transform(X_processed)
        
        return X_scaled
    
    def initialize_model(self, **kwargs) -> None:
        """Initialize the ML model based on model_type."""
        logger.info(f"Initializing {self.model_type} model...")
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 5),
                min_samples_leaf=kwargs.get('min_samples_leaf', 2),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Any]:
        """Train the model and return performance metrics."""
        logger.info("Training model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Preprocess features
        X_train_scaled = self.preprocess_features(X_train, fit_scaler=True)
        X_test_scaled = self.preprocess_features(X_test, fit_scaler=False)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"Training accuracy: {train_score:.4f}")
        logger.info(f"Test accuracy: {test_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Predictions and detailed metrics
        y_pred = self.model.predict(X_test_scaled)
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_scores': cv_scores,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def save_model(self, model_path: str) -> None:
        """Save the trained model and scaler."""
        logger.info(f"Saving model to {model_path}")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, model_path)
        logger.info("Model saved successfully!")


def main():
    parser = argparse.ArgumentParser(description='Train IoT threat detection model')
    parser.add_argument('--data_path', required=True, help='Path to preprocessed data CSV')
    parser.add_argument('--model_dest_path', required=True, help='Path to save trained model')
    parser.add_argument('--model_type', default='random_forest', 
                       choices=['random_forest', 'xgboost'],
                       help='Type of model to train')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = IoTThreatDetector(model_type=args.model_type)
    
    # Load data
    X, y = detector.load_data(args.data_path)
    
    # Initialize and train model
    detector.initialize_model()
    results = detector.train(X, y, test_size=args.test_size)
    
    # Print results
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    print(f"Model Type: {args.model_type}")
    print(f"Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"CV Score: {results['cv_scores'].mean():.4f} (+/- {results['cv_scores'].std() * 2:.4f})")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Save model
    detector.save_model(args.model_dest_path)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()