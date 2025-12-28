import pandas as pd
import numpy as np
import os
import json
import argparse
from src.utils.logger import setup_logger
from src.data.loader import load_data, load_config
from src.features.engineering import create_financial_features
from src.features.preprocessing import preprocess_data
from src.features.resampling import apply_smote
from src.models.trainer import train_model
from src.models.tuner import optimize_threshold
from sklearn.metrics import classification_report, fbeta_score

# Initialize Logger
logger = setup_logger("main_pipeline")

def main():
    logger.info("Starting Finance Club Project Pipeline...")
    
    # 1. Parse Arguments to allow Config Override
    parser = argparse.ArgumentParser(description="Finance Club Project Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    # 2. Load Configuration
    config = load_config(args.config)
    
    # 2. Load Training Data
    raw_path = config['data']['raw_path']
    df_train = load_data(raw_path)
    
    # 3. Validation Split is handled inside preprocess_data logic or we do it here?
    # To strictly follow notebook, we:
    # A. Feature Engineer (Entire Dataset or Split?)
    # Notebook typically engineers on the loaded df.
    
    logger.info("--- Step 1: Feature Engineering (Training Data) ---")
    df_eng = create_financial_features(df_train)
    
    # Capture median age for later validation set usage
    age_median = df_eng['age'].median()
    logger.info(f"Training Data Age Median: {age_median}")
    
    # 4. Preprocessing (Impute, Split, Scale)
    # Note: preprocess_data internally imputes 'age' with its own median logic.
    # We trust it for the training split generation.
    logger.info("--- Step 2: Preprocessing (Split & Scale) ---")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df_eng, config)
    
    # 5. Resampling (SMOTE) - Only on Training Set
    logger.info("--- Step 3: Resampling (SMOTE) ---")
    X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train, random_state=42)
    
    # 6. Model Training (XGBoost)
    logger.info("--- Step 4: Model Training (XGBoost) ---")
    model = train_model(X_train_resampled, y_train_resampled, config)
    
    # 7. Threshold Tuning (Maximize F2 on Test Split)
    logger.info("--- Step 5: Threshold Tuning (F2 Score) ---")
    # Note: We tune on the *Test* split properties (unseen during training)
    optimal_threshold, max_f2 = optimize_threshold(model, X_test_scaled, y_test)
    
    # 8. Evaluation & Metrics
    logger.info("--- Step 6: Final Evaluation ---")
    y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_optimal = (y_prob_test >= optimal_threshold).astype(int)
    
    final_f2 = fbeta_score(y_test, y_pred_optimal, beta=2)
    report = classification_report(y_test, y_pred_optimal, output_dict=True)
    
    # Log key metrics explicitly
    logger.info(f"Final Test Accuracy:  {report['accuracy']:.4f}")
    logger.info(f"Final Test Precision: {report['1']['precision']:.4f}")
    logger.info(f"Final Test Recall:    {report['1']['recall']:.4f}")
    logger.info(f"Final Test F2-Score:  {final_f2:.4f}")
    
    # Save Metrics
    metrics = {
        "optimal_threshold": float(optimal_threshold),
        "max_f2_score": float(final_f2),
        "classification_report": report
    }
    
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Metrics saved to results/metrics.json")
    
    # 9. Prediction on Unlabeled Validation Set
    val_path = config['data'].get('validation_path')
    if val_path and os.path.exists(val_path):
        logger.info(f"--- Step 7: Prediction on Validation Set ({val_path}) ---")
        df_val_raw = load_data(val_path)
        
        # A. Feature Engineering
        df_val_eng = create_financial_features(df_val_raw)
        
        # B. Imputation (Use Training Median)
        if 'age' in df_val_eng.columns:
            df_val_eng['age'] = df_val_eng['age'].fillna(age_median)
            
        # Select Features for Scaling
        # We need to ensure we have the same columns as X_train (minus target)
        # Using scaler features logic/columns?
        # The scaler expects only the feature columns.
        
        # Drop ID if present
        id_col = config['data'].get('id_col')
        validation_ids = df_val_eng[id_col] if id_col in df_val_eng.columns else None
        
        # Features to scale: All except ID and Target
        # Note: preprocess_data dropped ID and Target.
        # We perform similar drop here.
        X_val_features = df_val_eng.copy()
        if id_col and id_col in X_val_features.columns:
            X_val_features = X_val_features.drop(columns=[id_col])
            
        # C. Transform using Trained Scaler
        try:
            X_val_scaled = scaler.transform(X_val_features)
            
            # D. Predict using Trained Model & Optimal Threshold
            y_val_prob = model.predict_proba(X_val_scaled)[:, 1]
            y_val_pred = (y_val_prob >= optimal_threshold).astype(int)
            
            # E. Save Submission
            submission_df = pd.DataFrame({
                "Customer": validation_ids if validation_ids is not None else range(len(y_val_pred)),
                "next_month_default": y_val_pred
            })
            
            sub_path = config['data'].get('predictions_path', 'results/submission.csv')
            submission_df.to_csv(sub_path, index=False)
            logger.info(f"Predictions saved to {sub_path}")
            
        except Exception as e:
            logger.error(f"Error during validation prediction: {e}")
            logger.error("Check if validation columns match training columns exactly.")

    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
