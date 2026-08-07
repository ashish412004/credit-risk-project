import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load and prepare data (same as before)
file_path = 'accepted_loans.csv.csv'
df = pd.read_csv(file_path, low_memory=False)
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
df['default'] = df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)
df['emp_length'] = df['emp_length'].fillna('0')
df = df.dropna(subset=['dti', 'annual_inc', 'int_rate', 'loan_amnt'])
grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df['grade_num'] = df['grade'].map(grade_map)

# Features
features = ['grade_num', 'dti', 'loan_amnt']
X = df[features]
y = df['default']
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model fitting
model = sm.Logit(y_train, X_train).fit(disp=False)

# Predict on test set
y_pred_prob = model.predict(X_test)

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_prob)
print(f"\n================ MODEL PERFORMANCE ================")
print(f"ROC-AUC Score: {auc:.4f}")
print("Interpretation: 0.5 = Random Guessing, 0.7+ = Good, 0.8+ = Excellent")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve (Model Discrimination Power)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()