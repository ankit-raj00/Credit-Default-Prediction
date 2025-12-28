from xgboost import XGBClassifier
from src.utils.logger import setup_logger

logger = setup_logger("model_trainer")

def train_model(X_train, y_train, config):
    """
    Trains an XGBoost model using parameters from config.
    
    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training target.
        config (dict): Configuration dictionary containing model params.
        
    Returns:
        XGBClassifier: Trained model.
    """
    logger.info("Initializing XGBoost model...")
    
    params = config['model']['params']
    
    # Initialize model with unpacked parameters
    model = XGBClassifier(**params)
    
    logger.info(f"Training model with params: {params}...")
    model.fit(X_train, y_train)
    
    logger.info("Model training completed.")
    return model
