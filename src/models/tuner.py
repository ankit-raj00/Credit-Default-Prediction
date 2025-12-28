import numpy as np
from sklearn.metrics import fbeta_score
from src.utils.logger import setup_logger

logger = setup_logger("threshold_tuner")

def optimize_threshold(model, X_val, y_val):
    """
    Finds the optimal classification threshold to maximize F2-Score.
    
    Args:
        model: Trained classifier (must have predict_proba).
        X_val: Validation features.
        y_val: Validation true labels.
        
    Returns:
        tuple: optimal_threshold, max_f2_score
    """
    logger.info("Starting threshold optimization for F2-Score...")
    
    # Get probabilities for the positive class
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    thresholds = np.arange(0.01, 1.0, 0.005)
    f2_scores = []
    
    for t in thresholds:
        y_pred_t = (y_pred_proba >= t).astype(int)
        # Handle zero division implicitly or explicitly
        # sklearn's fbeta_score has zero_division parameter
        score = fbeta_score(y_val, y_pred_t, beta=2, zero_division=0)
        f2_scores.append(score)
        
    # Find max score and index
    max_score_idx = np.argmax(f2_scores)
    max_f2_score = f2_scores[max_score_idx]
    optimal_threshold = thresholds[max_score_idx]
    
    logger.info(f"Optimization complete. Optimal Threshold: {optimal_threshold:.3f}, Max F2-Score: {max_f2_score:.4f}")
    
    return optimal_threshold, max_f2_score, thresholds, f2_scores
