import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger("feature_engineering")

def create_financial_features(df):
    """
    Creates financial features 'AVG_Bill_amt' and 'PAY_TO_BILL_ratio'.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with new features.
    """
    logger.info("Creating financial features...")
    df_eng = df.copy()
    
    # Check if necessary columns exist
    bill_cols = [f'Bill_amt{i}' for i in range(1, 7)]
    pay_cols = [f'pay_amt{i}' for i in range(1, 7)]
    
    if not all(col in df.columns for col in bill_cols + pay_cols):
        logger.error("Missing necessary columns for feature engineering.")
        raise ValueError("Input DataFrame missing required Bill_amt or pay_amt columns.")

    # 1. AVG_Bill_amt: Average of Bill_amt1 to Bill_amt6
    df_eng['AVG_Bill_amt'] = df_eng[bill_cols].mean(axis=1)
    
    # 2. PAY_TO_BILL_ratio: Total Payment / Total Bill Amount
    # Sum of payments
    total_payment = df_eng[pay_cols].sum(axis=1)
    # Sum of bills
    total_bill = df_eng[bill_cols].sum(axis=1)
    
    # Handle division by zero or extremely small values mostly by replacing infinite with 0 or a cap
    # The notebook might have specific logic, but standard practice is to handle infs.
    # Looking at notebook description, it's "Ratio of total payment to total bill amount".
    # We will use a safe division or replace infs.
    
    # To replicate notebook behavior which resulted in specific values, we'll straight divide
    # and then handle the edge cases if any (notebook showed ratios like 0.03, 1.00, etc.)
    
    # Adding a small epsilon to avoid ZeroDivisionError if strict division is needed
    # But usually numpy handles this by returning inf.
    
    with np.errstate(divide='ignore', invalid='ignore'):
         ratio = total_payment / total_bill
    
    # Replace inf and -inf with NaN, then fill? Or matches notebook?
    # Notebook "PAY_TO_BILL_ratio" logic: "Ratio of total payment to total bill amount over 6 months."
    # We'll stick to a robust implementation:
    df_eng['PAY_TO_BILL_ratio'] = ratio
    
    # Basic cleanup of infinite/NaN ratios if total_bill was 0
    # Usually filling with 0 or 1 is strict business logic. 
    # If bill is 0 and payment is >0, ratio is inf. If both 0, ratio is NaN.
    # We will fill NaNs with 0 (no bill, no payment ratio logic) and infs with a high cap or 0?
    # Given notebook didn't specify strict cleaning here, we will fillna(0) and replace infs with 0 for safety for now,
    # or better, check if the notebook had cleaning for this.
    # Notebook analysis showed min -546.93 and max 5.04 (line 2395 of view_file). 
    # This implies no capping was done or it was done loosely. 
    # We will just fillNa(0) for safety.
    
    df_eng['PAY_TO_BILL_ratio'] = df_eng['PAY_TO_BILL_ratio'].replace([np.inf, -np.inf], 0).fillna(0)

    logger.info("Financial features created.")
    return df_eng
