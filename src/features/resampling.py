from imblearn.over_sampling import SMOTE
from src.utils.logger import setup_logger

logger = setup_logger("resampling")

def apply_smote(X_train, y_train, random_state=42):
    """
    Applies SMOTE to the training data to handle class imbalance.
    
    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training target.
        random_state (int): Random seed.
        
    Returns:
        tuple: X_resampled, y_resampled
    """
    logger.info(f"Applying SMOTE with random_state={random_state}...")
    
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    logger.info(f"SMOTE applied. Original shape: {X_train.shape}, Resampled shape: {X_resampled.shape}")
    
    return X_resampled, y_resampled
