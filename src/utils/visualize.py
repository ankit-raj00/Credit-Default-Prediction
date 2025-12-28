import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import pandas as pd
import numpy as np

def save_plot(fig, filename, output_dir):
    """Saves a matplotlib figure to the specified directory."""
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)

def plot_f2_vs_threshold(y_test, y_pred_proba, thresholds, f2_scores, optimal_threshold, max_f2_score, output_dir):
    """Plots F2-Score vs Threshold curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f2_scores, marker='o', linestyle='-', markersize=2, label='F2-Score')
    ax.set_title('F2-Score vs. Classification Threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F2-Score')
    ax.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal: {optimal_threshold:.3f} (F2={max_f2_score:.4f})')
    ax.grid(True)
    ax.legend()
    save_plot(fig, 'f2_score_vs_threshold.png', output_dir)

def plot_roc_curve(y_test, y_pred_proba, output_dir):
    """Plots ROC Curve."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc="lower right")
    ax.grid(True)
    save_plot(fig, 'roc_curve.png', output_dir)

def plot_precision_recall_curve(y_test, y_pred_proba, output_dir):
    """Plots Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="best")
    ax.grid(True)
    save_plot(fig, 'precision_recall_curve.png', output_dir)

def plot_confusion_matrix(y_test, y_pred, output_dir):
    """Plots Confusion Matrix."""
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Default', 'Default'], 
                yticklabels=['No Default', 'Default'])
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    save_plot(fig, 'confusion_matrix.png', output_dir)

def plot_feature_importance(model, feature_names, output_dir):
    """Plots Feature Importance."""
    # Handle different model types (XGBoost vs others)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        # If feature_names provided
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
            
        # Create dataframe
        df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
        df_imp = df_imp.sort_values('importance', ascending=False).head(20) # Top 20
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=df_imp, palette='viridis')
        ax.set_title('Top 20 Feature Importance')
        save_plot(fig, 'feature_importance.png', output_dir)
