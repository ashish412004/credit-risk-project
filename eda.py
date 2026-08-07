import pandas as pd
import numpy as np

print("1. Loading dataset...")
# Load the cleaned/raw file correctly
file_path = 'accepted_loans.csv.csv'
df = pd.read_csv(file_path, low_memory=False)

print(f"Dataset loaded! Total rows: {len(df)}")

# 2. Filter for finished loans (Fully Paid vs Charged Off)
target_statuses = ['Fully Paid', 'Charged Off']
df = df[df['loan_status'].isin(target_statuses)]

# 3. Create Target Variable: Default = 1, Paid = 0
df['default'] = df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)

# 4. Remove Data Leakage Columns
leakage_cols = ['total_pymnt', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt']
df = df.drop(columns=[col for col in leakage_cols if col in df.columns])

# 5. Handle Nulls (Banking logic)
df['emp_length'] = df['emp_length'].fillna('0')
df = df.dropna(subset=['dti', 'annual_inc', 'int_rate'])

print(f"Data cleaned. Active rows after cleaning: {len(df)}")

print("\n2. Engineering Risk Features...")
# Feature 1: Income to Loan Ratio (Ability to pay)
df['inc_to_loan'] = df['annual_inc'] / df['loan_amnt']

# Feature 2: Employment Stability Flag (Longer tenure = lower risk)
df['emp_stability'] = df['emp_length'].apply(lambda x: 1 if x in ['10+ years', '9 years', '8 years'] else 0)

# Feature 3: High DTI Flag (Threshold strategy > 25%)
df['high_dti'] = df['dti'].apply(lambda x: 1 if x > 25 else 0)

# Feature 4: Grade Numeric Mapping (A=1 to G=7 for modeling)
grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df['grade_num'] = df['grade'].map(grade_map)

# Feature 5: Debt Consolidation Flag
df['is_debt_consolidation'] = df['purpose'].apply(lambda x: 1 if x == 'debt_consolidation' else 0)

print("Feature Engineering Completed Successfully!")
print("Newly created columns: inc_to_loan, emp_stability, high_dti, grade_num, is_debt_consolidation")
print(df[['annual_inc', 'loan_amnt', 'inc_to_loan', 'grade', 'grade_num']].head(3))