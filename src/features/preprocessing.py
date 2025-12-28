import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils.logger import setup_logger

logger = setup_logger("preprocessing")

def preprocess_data(df, config):
    """
    Performs data cleaning (imputation), splitting, and scaling.
    
    Args:
        df (pd.DataFrame): Input DataFrame (after feature engineering).
        config (dict): Configuration dictionary.
        
    Returns:
        tuple: X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    logger.info("Starting preprocessing...")
    
    target_col = config['data']['target_col']
    test_size = 0.2 # Standard default, or check config
    # Notebook used 20%? Or 15%? Config didn't specify split ratio, we'll default to 0.2 (common).
    
    # 1. Imputation
    # Notebook: "Missing 'age' values ... imputed with median: 34.0"
    if 'age' in df.columns and df['age'].isnull().sum() > 0:
        fill_val = df['age'].median()
        df['age'] = df['age'].fillna(fill_val)
        logger.info(f"Imputed missing 'age' values with median: {fill_val}")
        
    # 2. X/y Split
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        # Drop ID column if present as it's not a feature
        id_col = config['data'].get('id_col')
        if id_col and id_col in X.columns:
            X = X.drop(columns=[id_col])
    else:
        # Validation/Test scenario logic handled separately usually, but here we assume training flow
        logger.warning("Target column not found. Assuming inference mode or strictly X.")
        X = df.copy()
        y = None
        id_col = config['data'].get('id_col')
        if id_col and id_col in X.columns:
             X = X.drop(columns=[id_col])

    # 3. Train/Test Split
    # We must split BEFORE scaling to avoid leakage
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Split data: Train shape {X_train.shape}, Test shape {X_test.shape}")
    else:
        # If no target, simply scale everything (Verification/Submission mode)
        X_train = X 
        X_test = None
        y_train = None
        y_test = None

    # 4. Scaling
    # Notebook uses StandardScaler
    scaler = StandardScaler()
    
    # Fit on TRAIN, transform BOTH
    X_train_scaled = scaler.fit_transform(X_train)
    # Convert back to DF for convenience if needed, but array is standard for sklearn
    # Keeping as array for now, or X_train is dataframe? 
    # XGBoost handles arrays fine.
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = None
        
    logger.info("Data scaling completed.")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
