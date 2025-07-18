#!/usr/bin/env python3
"""
Model evaluation module for IoT Network Threat Detection.
Provides comprehensive evaluation metrics and visualization tools.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IoTThreatEvaluator:
    """Evaluator for IoT threat detection models."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.model_type = None
        
    def load_model(self) -> None:
        """Load the trained model and associated components."""
        logger.info(f"Loading model from {self.model_path}")
        
        self.model_data = joblib.load(self.model_path)
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.feature_columns = self.model_data['feature_columns']
        self.model_type = self.model_data['model_type']
        
        logger.info(f"Loaded {self.model_type} model with {len(self.feature_columns)} features")
    
    def load_test_data(self, test_data_path: str) -> tuple:
        """Load test data for evaluation."""
        logger.info(f"Loading test data from {test_data_path}")
        
        df = pd.read_csv(test_data_path)
        
        # Assume the target column is 'label_encoded' or 'label'
        target_col = 'label_encoded' if 'label_encoded' in df.columns else 'label'
        
        if target_col not in df.columns:
            raise ValueError(f"No target column found. Expected 'label' or 'label_encoded'")
        
        # Separate features and target
        y = df[target_col]
        X = df.drop([target_col, 'label'], axis=1, errors='ignore')
        
        # Ensure feature columns match training data
        if list(X.columns) != self.feature_columns:
            logger.warning("Feature columns don't match training data. Attempting to align...")
            X = X[self.feature_columns]
        
        logger.info(f"Loaded {len(X)} test samples")
        return X, y
    
    def preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features using the trained scaler."""
        logger.info("Preprocessing features...")
        
        # Handle categorical features if any
        X_processed = X.copy()
        
        # Convert categorical columns to numeric
        for col in X_processed.select_dtypes(include=['object']).columns:
            X_processed[col] = pd.Categorical(X_processed[col]).codes
        
        # Scale features
        X_scaled = self.scaler.transform(X_processed)
        
        return X_scaled
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        logger.info("Evaluating model...")
        
        # Preprocess features
        X_scaled = self.preprocess_features(X)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)
        
        # Basic metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='weighted')
        
        # Classification report
        class_report = classification_report(y, y_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)
        
        # ROC-AUC for binary/multiclass
        try:
            if len(np.unique(y)) == 2:
                # Binary classification
                roc_auc = roc_auc_score(y, y_proba[:, 1])
            else:
                # Multiclass classification
                roc_auc = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            roc_auc = None
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        logger.info(f"Evaluation complete. Accuracy: {accuracy:.4f}")
        return results
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, class_names: Optional[list] = None,
                             save_path: Optional[str] = None) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(conf_matrix))]
        
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                      save_path: Optional[str] = None) -> None:
        """Plot ROC curve for binary classification."""
        if len(np.unique(y_true)) != 2:
            logger.warning("ROC curve plotting only supported for binary classification")
            return
        
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"ROC curve saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report."""
        report = f"""
IoT Network Threat Detection - Model Evaluation Report
====================================================

Model Type: {self.model_type}
Model Path: {self.model_path}

Performance Metrics:
------------------
Accuracy:  {results['accuracy']:.4f}
Precision: {results['precision']:.4f}
Recall:    {results['recall']:.4f}
F1-Score:  {results['f1_score']:.4f}
ROC-AUC:   {results['roc_auc']:.4f if results['roc_auc'] else 'N/A'}

Detailed Classification Report:
-----------------------------
"""
        
        # Add per-class metrics
        for class_name, metrics in results['classification_report'].items():
            if isinstance(metrics, dict):
                report += f"\nClass {class_name}:\n"
                report += f"  Precision: {metrics['precision']:.4f}\n"
                report += f"  Recall:    {metrics['recall']:.4f}\n"
                report += f"  F1-Score:  {metrics['f1-score']:.4f}\n"
                report += f"  Support:   {metrics['support']}\n"
        
        report += f"\nConfusion Matrix:\n{results['confusion_matrix']}\n"
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Evaluate IoT threat detection model')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--test_data_path', required=True, help='Path to test data CSV')
    parser.add_argument('--output_dir', default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save evaluation plots')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = IoTThreatEvaluator(args.model_path)
    evaluator.load_model()
    
    # Load test data
    X_test, y_test = evaluator.load_test_data(args.test_data_path)
    
    # Evaluate model
    results = evaluator.evaluate_model(X_test, y_test)
    
    # Generate and print report
    report = evaluator.generate_report(results)
    print(report)
    
    # Save report
    report_path = Path(args.output_dir) / 'evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Evaluation report saved to {report_path}")
    
    # Generate and save plots
    if args.save_plots:
        # Confusion matrix
        cm_path = Path(args.output_dir) / 'confusion_matrix.png'
        evaluator.plot_confusion_matrix(results['confusion_matrix'], save_path=cm_path)
        
        # ROC curve (if binary classification)
        if len(np.unique(y_test)) == 2:
            roc_path = Path(args.output_dir) / 'roc_curve.png'
            evaluator.plot_roc_curve(y_test, results['probabilities'], save_path=roc_path)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()