import pandas as pd

# Load data
df = pd.read_csv("data/loan_data.csv")
# print(df.head())
# Check missing values
# print(df.isnull().sum())
# print(df['Gender'].value_counts())

# Gender column
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0]).astype('int')

# Married column
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0}).astype('int')

# Dependents column
def validate_dependents(x):
    if pd.isnull(x):
        return None  # leave NaN for now
    if x > 3:
        return 3
    elif x < 0:
        return 0
    else:
        return x  # IMPORTANT: return x for normal values

df['Dependents'] = df['Dependents'].replace('3+', 3)
df['Dependents'] = df['Dependents'].astype('float64')
df['Dependents'] = df['Dependents'].map(validate_dependents)
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Dependents'] = df['Dependents'].astype('int')

# Education column
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Education'] = df['Education'].astype('int')

# Self_Employed column
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})

# ApplicantIncome
# print(df['ApplicantIncome'].head(10))
# print(df['ApplicantIncome'].isna().sum())

# 

print(df.info())