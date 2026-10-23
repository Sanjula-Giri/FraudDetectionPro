import pandas as pd
import xgboost as xgb
import mlflow
from sklearn.metrics import precision_recall_curve, auc

train = pd.read_csv(r'E:\FraudDetectionProject\data\processed\train.csv')
val = pd.read_csv(r'E:\FraudDetectionProject\data\processed\val.csv')

X_train, y_train = train.drop(columns=['Class', 'Time']), train['Class']
X_val, y_val = val.drop(columns=['Class', 'Time']), val['Class']

# Calculate weight for the minority class
imbalance_weight = (len(y_train) - sum(y_train)) / sum(y_train)

mlflow.set_experiment("Fraud_Detection_XGBoost")
with mlflow.start_run():
    model = xgb.XGBClassifier(scale_pos_weight=imbalance_weight, eval_metric="aucpr", early_stopping_rounds=10)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    y_val_probs = model.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, y_val_probs)
    pr_auc = auc(recall, precision)
    
    mlflow.log_metric("pr_auc", pr_auc)
    print(f"XGBoost PR-AUC: {pr_auc:.4f}")
    import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# 1. Calculate Recall at 1% False Positive Rate (FPR)
fpr, tpr, thresholds = roc_curve(y_val, y_val_probs)
idx_1_fpr = np.where(fpr <= 0.01)[0][-1]
recall_at_1 = tpr[idx_1_fpr]
print(f"Recall at 1% FPR: {recall_at_1:.4f}")

# 2. Cost-Based Threshold Optimizer
# Assumes missing a fraud costs $1000, and a manual review costs $10
P = sum(y_val == 1)
N = sum(y_val == 0)
TP = tpr * P
FN = P - TP
FP = fpr * N

costs = (FN * 1000) + (FP * 10)
best_threshold_idx = np.argmin(costs)
best_threshold = thresholds[best_threshold_idx]
print(f"Optimal Cost-Based Threshold: {best_threshold:.4f}")

# 3. Generate SHAP Explanations
print("Generating SHAP summary plot...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

plt.figure()
shap.summary_plot(shap_values, X_val, show=False)
plt.savefig(r'E:\FraudDetectionProject\notebooks\shap_summary.png', bbox_inches='tight')
print("SHAP plot saved to E:\\FraudDetectionProject\\notebooks\\shap_summary.png")

import joblib
model.save_model(r'E:\FraudDetectionProject\src\models\xgb_model.json')
joblib.dump(explainer, r'E:\FraudDetectionProject\src\models\shap_explainer.pkl')
print("Model and explainer saved successfully.")