import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils.outlier import define_outlier, cap_outlier

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

# CoapplicantIncome column
# print(df['CoapplicantIncome'].head(10))
# print(df['CoapplicantIncome'].isna().sum())
df['CoapplicantIncome'] = df['CoapplicantIncome'].astype('float')

# LoanAmount column
# print(df['LoanAmount'].head(10))
# print(df['LoanAmount'].isna().sum())

# Loan_Amount_Term column
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])

# Credit_History column
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# Property_Area column
df['Property_Area'] = df['Property_Area'].map({'Rural': 0, 'Urban': 1, 'Semiurban': 2})

# Loan_Status column
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Check for duplicates
# duplicates = df[df.duplicated()]
# print(duplicates)
# print(df.duplicated().sum())

# Apply log transformation to ApplicationIncome
df['ApplicantIncome_log'] = np.log(df['ApplicantIncome'] + 1)

# Perform scaling to ApplicantIncome_log
ApplicantIncome_scaler = StandardScaler()
df['ApplicantIncome_scale'] = ApplicantIncome_scaler.fit_transform(df[['ApplicantIncome_log']])

# CoapplicantIncome
df['CoapplicantIncome'] = df['CoapplicantIncome'].round(2)
# lower, upper = define_outlier(df, 'CoapplicantIncome')
# print(f'CoapplicantIncome outlier\nlower: {lower} and upper: {upper}')
df['CoapplicantIncome_cap'] = cap_outlier(df, 'CoapplicantIncome')

df['CoapplicantIncome_cap_log'] = np.log(df['CoapplicantIncome_cap'] + 1)
CoapplicantIncome_scaler = StandardScaler()
df['CoapplicantIncome_cap_scale'] = CoapplicantIncome_scaler.fit_transform(df[['CoapplicantIncome_cap_log']])

# LoanAmount
lower, upper = define_outlier(df, 'LoanAmount')
print(f'LoanAmount outlier\nlower: {lower} and upper: {upper}')
print('lower:', df['LoanAmount'].loc[df['LoanAmount'] < lower])
print('upper:', df['LoanAmount'].loc[df['LoanAmount'] > upper])

plt.boxplot(df['LoanAmount'], vert=False)
plt.title("LoanAmount")
plt.show()