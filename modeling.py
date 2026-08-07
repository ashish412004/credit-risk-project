import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

print("1. Loading dataset for risk modeling...")
file_path = 'accepted_loans.csv.csv'
df = pd.read_csv(file_path, low_memory=False)

# Filter and target setup
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
df['default'] = df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)

# Feature engineering preparation
df['emp_length'] = df['emp_length'].fillna('0')
df = df.dropna(subset=['dti', 'annual_inc', 'int_rate', 'loan_amnt'])
df['inc_to_loan'] = df['annual_inc'] / df['loan_amnt']
df['emp_stability'] = df['emp_length'].apply(lambda x: 1 if x in ['10+ years', '9 years', '8 years'] else 0)
df['high_dti'] = df['dti'].apply(lambda x: 1 if x > 25 else 0)
grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df['grade_num'] = df['grade'].map(grade_map)

# Take a robust sample of 100,000 rows for smooth model training
df_sample = df.sample(n=100000, random_state=42)

# Select features for risk prediction
features = ['grade_num', 'dti', 'inc_to_loan', 'emp_stability', 'high_dti', 'loan_amnt']
X = df_sample[features]
y = df_sample['default']

# Add constant for statsmodels intercept
X = sm.add_constant(X)

# Train/Test Split (80/20 stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("2. Fitting Logistic Regression model (statsmodels)...")
logit_model = sm.Logit(y_train, X_train)
result = logit_model.fit(disp=False)

print("\n================ MODEL SUMMARY ================")
print(result.summary())

print("\n================ ODDS RATIOS (Risk Impact) ================")
odds_ratios = np.exp(result.params)
print(odds_ratios)