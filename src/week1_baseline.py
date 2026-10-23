import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc

# Load the processed splits
train = pd.read_csv(r'E:\FraudDetectionProject\data\processed\train.csv')
val = pd.read_csv(r'E:\FraudDetectionProject\data\processed\val.csv')

# Separate features (X) and target (y)
# We drop 'Time' as it is generally not a direct predictive feature for the baseline
X_train = train.drop(columns=['Class', 'Time'])
y_train = train['Class']
X_val = val.drop(columns=['Class', 'Time'])
y_val = val['Class']

# Initialize and train the logistic regression baseline
print("Training baseline model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict probabilities on the validation set
y_val_probs = model.predict_proba(X_val)[:, 1]

# Calculate PR-AUC
precision, recall, _ = precision_recall_curve(y_val, y_val_probs)
pr_auc = auc(recall, precision)

print(f"Baseline PR-AUC: {pr_auc:.4f}")