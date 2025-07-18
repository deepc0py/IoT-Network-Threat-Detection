#!/usr/bin/env python3
"""
Data preprocessing module for IoT Network Threat Detection.
Handles data cleaning, feature engineering, and dataset preparation.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IoTDataPreprocessor:
    """Preprocessor for IoT network traffic data."""
    
    def __init__(self):
        self.label_mapping = {}
        self.feature_columns = []
        
    def load_data(self, input_path: str) -> pd.DataFrame:
        """Load raw IoT network data from CSV."""
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        logger.info("Cleaning data...")
        
        # Handle missing values
        df = df.dropna()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        logger.info(f"After cleaning: {len(df)} records remaining")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for network traffic analysis."""
        logger.info("Engineering features...")
        
        # Feature engineering will be implemented based on dataset structure
        # Placeholder for now
        
        return df
    
    def encode_labels(self, df: pd.DataFrame, label_column: str = 'label') -> pd.DataFrame:
        """Encode categorical labels for classification."""
        if label_column in df.columns:
            logger.info(f"Encoding labels in column: {label_column}")
            # Store label mapping for later use
            unique_labels = df[label_column].unique()
            self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            df[f'{label_column}_encoded'] = df[label_column].map(self.label_mapping)
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """Save preprocessed data to CSV."""
        logger.info(f"Saving processed data to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} records")


def main():
    parser = argparse.ArgumentParser(description='Preprocess IoT network data')
    parser.add_argument('--input_path', required=True, help='Path to input CSV file')
    parser.add_argument('--output_path', required=True, help='Path to output CSV file')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = IoTDataPreprocessor()
    
    # Load and process data
    df = preprocessor.load_data(args.input_path)
    df = preprocessor.clean_data(df)
    df = preprocessor.engineer_features(df)
    df = preprocessor.encode_labels(df)
    
    # Save processed data
    preprocessor.save_processed_data(df, args.output_path)
    
    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()