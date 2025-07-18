#!/usr/bin/env python3
"""
Data preprocessing module for IoT Network Threat Detection.
Handles data cleaning, feature engineering, and dataset preparation.
Based on EDA findings from TON_IoT dataset analysis.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IoTDataPreprocessor:
    """Preprocessor for IoT network traffic data."""
    
    def __init__(self, dataset_type: str = 'auto'):
        """
        Initialize preprocessor.
        
        Args:
            dataset_type: 'iot', 'network', or 'auto' for automatic detection
        """
        self.dataset_type = dataset_type
        self.label_mapping = {}
        self.feature_columns = []
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.class_weights = {}
        
    def load_data(self, input_path: str) -> pd.DataFrame:
        """Load raw IoT network data from CSV."""
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path, low_memory=False)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        
        # Auto-detect dataset type
        if self.dataset_type == 'auto':
            if 'current_temperature' in df.columns:
                self.dataset_type = 'iot'
                logger.info("Detected IoT device dataset")
            elif 'src_ip' in df.columns or 'dst_ip' in df.columns:
                self.dataset_type = 'network'
                logger.info("Detected Network traffic dataset")
            else:
                self.dataset_type = 'unknown'
                logger.warning("Could not auto-detect dataset type")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        logger.info("Cleaning data...")
        original_length = len(df)
        
        # Handle missing values
        if df.isnull().sum().sum() > 0:
            logger.info(f"Found {df.isnull().sum().sum()} missing values")
            
            # For IoT datasets, drop rows with missing temporal data
            if self.dataset_type == 'iot':
                df = df.dropna(subset=['current_temperature', 'thermostat_status'])
            
            # For other missing values, use forward fill or drop
            df = df.fillna(method='ffill').fillna(method='bfill')
            df = df.dropna()
        
        # Remove duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.info(f"Removing {duplicates} duplicate records")
            df = df.drop_duplicates()
        
        logger.info(f"Cleaned data: {len(df)} records (removed {original_length - len(df)} records)")
        return df
    
    def engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer temporal features from date/time columns."""
        if self.dataset_type == 'iot' and 'date' in df.columns and 'time' in df.columns:
            logger.info("Engineering temporal features...")
            
            # Combine date and time, handle missing values
            valid_datetime_mask = df['date'].notna() & df['time'].notna()
            
            if valid_datetime_mask.sum() > 0:
                # Parse datetime for valid entries
                df_valid = df[valid_datetime_mask].copy()
                datetime_str = df_valid['date'].astype(str) + ' ' + df_valid['time'].astype(str)
                
                try:
                    dt = pd.to_datetime(datetime_str, format='%d-%b-%y %H:%M:%S', errors='coerce')
                    
                    # Extract temporal features
                    df.loc[valid_datetime_mask, 'hour'] = dt.dt.hour
                    df.loc[valid_datetime_mask, 'day_of_week'] = dt.dt.dayofweek
                    df.loc[valid_datetime_mask, 'day_of_month'] = dt.dt.day
                    df.loc[valid_datetime_mask, 'month'] = dt.dt.month
                    df.loc[valid_datetime_mask, 'is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
                    
                    # Time-based features
                    df.loc[valid_datetime_mask, 'is_business_hours'] = ((dt.dt.hour >= 9) & (dt.dt.hour <= 17)).astype(int)
                    df.loc[valid_datetime_mask, 'is_night'] = ((dt.dt.hour >= 22) | (dt.dt.hour <= 6)).astype(int)
                    
                    logger.info("Successfully engineered temporal features")
                    
                except Exception as e:
                    logger.warning(f"Could not parse datetime: {e}")
                    # Fill with default values
                    df['hour'] = 12
                    df['day_of_week'] = 1
                    df['day_of_month'] = 15
                    df['month'] = 6
                    df['is_weekend'] = 0
                    df['is_business_hours'] = 1
                    df['is_night'] = 0
            else:
                logger.warning("No valid datetime entries found")
                
        elif self.dataset_type == 'network' and 'ts' in df.columns:
            logger.info("Engineering network temporal features...")
            
            # Handle network timestamps
            try:
                df['ts'] = pd.to_datetime(df['ts'], unit='s', errors='coerce')
                df['hour'] = df['ts'].dt.hour
                df['day_of_week'] = df['ts'].dt.dayofweek
                df['is_weekend'] = (df['ts'].dt.dayofweek >= 5).astype(int)
                df['is_business_hours'] = ((df['ts'].dt.hour >= 9) & (df['ts'].dt.hour <= 17)).astype(int)
                
            except Exception as e:
                logger.warning(f"Could not parse network timestamps: {e}")
        
        return df
    
    def engineer_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer network-specific features."""
        if self.dataset_type == 'network':
            logger.info("Engineering network features...")
            
            # Convert numeric columns to proper types
            numeric_cols = ['src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts', 'duration', 'src_port', 'dst_port']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Byte ratios
            if 'src_bytes' in df.columns and 'dst_bytes' in df.columns:
                df['byte_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)  # +1 to avoid division by zero
                df['total_bytes'] = df['src_bytes'] + df['dst_bytes']
            
            # Packet ratios
            if 'src_pkts' in df.columns and 'dst_pkts' in df.columns:
                df['pkt_ratio'] = df['src_pkts'] / (df['dst_pkts'] + 1)
                df['total_pkts'] = df['src_pkts'] + df['dst_pkts']
            
            # Duration-based features
            if 'duration' in df.columns:
                df['duration_log'] = np.log1p(df['duration'])  # log(1+x) to handle zeros
                df['is_short_duration'] = (df['duration'] < 1).astype(int)
                df['is_long_duration'] = (df['duration'] > 3600).astype(int)  # > 1 hour
            
            # Port-based features
            if 'src_port' in df.columns:
                df['src_port_is_system'] = (df['src_port'] < 1024).astype(int)
                df['src_port_is_ephemeral'] = (df['src_port'] > 32767).astype(int)
            
            if 'dst_port' in df.columns:
                df['dst_port_is_system'] = (df['dst_port'] < 1024).astype(int)
                df['dst_port_is_ephemeral'] = (df['dst_port'] > 32767).astype(int)
            
            # Common ports
            common_ports = [80, 443, 53, 22, 21, 25, 110, 993, 995]
            if 'dst_port' in df.columns:
                df['dst_port_is_common'] = df['dst_port'].isin(common_ports).astype(int)
            
            logger.info("Completed network feature engineering")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        logger.info("Encoding categorical features...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['label', 'type']]
        
        for col in categorical_cols:
            if col in df.columns:
                # Use label encoding for high cardinality features
                if df[col].nunique() > 20:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                    else:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
                else:
                    # Use one-hot encoding for low cardinality features
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
        
        return df
    
    def prepare_features_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and labels."""
        logger.info("Preparing features and labels...")
        
        # Determine target column
        if 'label' in df.columns:
            target_col = 'label'
        else:
            raise ValueError("No target column found. Expected 'label' column.")
        
        # Separate features and target
        y = df[target_col]
        
        # Drop non-feature columns
        columns_to_drop = [target_col, 'type', 'date', 'time', 'ts', 'src_ip', 'dst_ip']
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        
        X = df.drop(columns=columns_to_drop)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        logger.info(f"Prepared {len(X.columns)} features for {len(X)} samples")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def scale_features(self, X: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        logger.info("Scaling features...")
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled
    
    def compute_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """Compute class weights for imbalanced datasets."""
        logger.info("Computing class weights for imbalanced data...")
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        self.class_weights = {cls: weight for cls, weight in zip(classes, weights)}
        
        logger.info(f"Class weights: {self.class_weights}")
        return self.class_weights
    
    def process_data(self, input_path: str, output_path: str, 
                    test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """Complete data processing pipeline."""
        logger.info("Starting complete data processing pipeline...")
        
        # Load data
        df = self.load_data(input_path)
        
        # Clean data
        df = self.clean_data(df)
        
        # Feature engineering
        df = self.engineer_temporal_features(df)
        df = self.engineer_network_features(df)
        df = self.encode_categorical_features(df)
        
        # Prepare features and labels
        X, y = self.prepare_features_labels(df)
        
        # Scale features
        X_scaled = self.scale_features(X, fit_scaler=True)
        
        # Compute class weights
        class_weights = self.compute_class_weights(y)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, 
            stratify=y if len(y.unique()) > 1 else None
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Save processed data
        self.save_processed_data(X_train, X_test, y_train, y_test, output_path)
        
        # Save preprocessing artifacts
        self.save_preprocessing_artifacts(output_path)
        
        results = {
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'feature_columns': self.feature_columns,
            'class_weights': class_weights,
            'dataset_type': self.dataset_type
        }
        
        logger.info("Data processing pipeline completed successfully!")
        return results
    
    def save_processed_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series, output_path: str) -> None:
        """Save processed data to files."""
        logger.info(f"Saving processed data to {output_path}")
        
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save train/test splits
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_path = output_dir / 'train_data.csv'
        test_path = output_dir / 'test_data.csv'
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Saved train data: {train_path}")
        logger.info(f"Saved test data: {test_path}")
    
    def save_preprocessing_artifacts(self, output_path: str) -> None:
        """Save preprocessing artifacts (scaler, encoders, etc.)."""
        output_dir = Path(output_path).parent
        artifacts_path = output_dir / 'preprocessing_artifacts.pkl'
        
        artifacts = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'class_weights': self.class_weights,
            'dataset_type': self.dataset_type
        }
        
        joblib.dump(artifacts, artifacts_path)
        logger.info(f"Saved preprocessing artifacts: {artifacts_path}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess IoT network data')
    parser.add_argument('--input_path', required=True, help='Path to input CSV file')
    parser.add_argument('--output_path', required=True, help='Path to output directory')
    parser.add_argument('--dataset_type', default='auto', 
                       choices=['auto', 'iot', 'network'],
                       help='Type of dataset to process')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = IoTDataPreprocessor(dataset_type=args.dataset_type)
    
    # Process data
    results = preprocessor.process_data(
        input_path=args.input_path,
        output_path=args.output_path,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Print results
    print("\n" + "="*50)
    print("PREPROCESSING RESULTS")
    print("="*50)
    print(f"Dataset Type: {results['dataset_type']}")
    print(f"Training Set Shape: {results['train_shape']}")
    print(f"Test Set Shape: {results['test_shape']}")
    print(f"Number of Features: {len(results['feature_columns'])}")
    print(f"Class Weights: {results['class_weights']}")
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()