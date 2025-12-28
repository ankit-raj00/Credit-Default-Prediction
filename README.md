# Finance Club Project - Credit Default Prediction

## Overview
This project implements a modular Machine Learning pipeline to predict credit card defaults (`next_month_default`) using **XGBoost**. The model is specifically optimized for the **F2-Score** to prioritize Recall (capturing more defaulters) while maintaining reasonable precision.

## Key Features
- **Modular Architecture**: Clean separation of Data, Features, Models, and Configuration.
- **Config-Driven**: All hyperparameters and paths are defined in `config/config.yaml`.
- **F2-Score Optimization**: Classification thresholds are dynamically tuned to maximize F2-Score (Optimal ~0.325).
- **Financial Feature Engineering**: Includes calculated features like `AVG_Bill_amt` and `PAY_TO_BILL_ratio`.
- **Imbalance Handling**: Uses **SMOTE** (Synthetic Minority Over-sampling) on training data.
- **Logging & Metrics**: Centralized logging to `logs/` and metric validation to `results/metrics.json`.

## Project Structure
```
FinanceClub_Project/
├── config/
│   └── config.yaml          # Hyperparameters & Paths
├── data/
│   └── raw/                 # Place 'train_dataset_final1.csv' & 'validate_dataset_final.csv' here
├── logs/                    # Execution logs
├── results/                 # Metrics and Submission CSV
├── src/
│   ├── data/
│   ├── features/
│   │   ├── engineering.py   # Financial features
│   │   ├── preprocessing.py # Scaling, Imputation, Splitting
│   │   └── resampling.py    # SMOTE
│   ├── models/
│   │   ├── trainer.py       # XGBoost Training
│   │   └── tuner.py         # Threshold Optimization
│   └── utils/
│       └── logger.py
├── main.py                  # Pipeline Entry Point
└── requirements.txt         # Dependencies
```

## Setup & Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   Ensure `train_dataset_final1.csv` and `validate_dataset_final.csv` are in `data/raw/`.

3. **Run Pipeline**:
   ```bash
   python main.py
   ```

4. **Output**:
   - Metrics: `results/metrics.json`
   - Predictions: `results/submission.csv`
   - Logs: `logs/pipeline_YYYYMMDD_HHMMSS.log`

## Model Details
- **Algorithm**: XGBoost Classifier
- **Optimization Metric**: F2-Score
- **Hyperparameters**: Tuned for high recall (`scale_pos_weight: ~6.38`, `max_depth: 7`).
