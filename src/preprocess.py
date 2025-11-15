import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Identify Outliers
# ApplicantIncome_q1 = df['ApplicantIncome'].quantile(.25)
# ApplicantIncome_q3 = df['ApplicantIncome'].quantile(.75)
# ApplicantIncome_iqr = ApplicantIncome_q3 - ApplicantIncome_q1
# ApplicantIncome_lower = ApplicantIncome_q1 - (1.5 * ApplicantIncome_iqr)
# ApplicantIncome_upper = ApplicantIncome_q3 + (1.5 * ApplicantIncome_iqr)

# plt.boxplot(df['ApplicantIncome'])
# plt.title("Applicant Income")
# plt.show()

# CoapplicantIncome_q1 = df['CoapplicantIncome'].quantile(.25)
# CoapplicantIncome_q3 = df['CoapplicantIncome'].quantile(.75)
# CoapplicantIncome_iqr = CoapplicantIncome_q3 - CoapplicantIncome_q1
# CoapplicantIncome_lower = CoapplicantIncome_q1 - (1.5 * CoapplicantIncome_iqr)
# CoapplicantIncome_upper = CoapplicantIncome_q3 + (1.5 * CoapplicantIncome_iqr)

# LoanAmount_q1 = df['LoanAmount'].quantile(.25)
# LoanAmount_q3 = df['LoanAmount'].quantile(.75)
# LoanAmount_iqr = LoanAmount_q3 - LoanAmount_q1
# LoanAmount_lower = LoanAmount_q1 - (1.5 * LoanAmount_iqr)
# LoanAmount_upper = LoanAmount_q3 + (1.5 * LoanAmount_iqr)


# Apply log transformation to ApplicationIncome
# fig, ax = plt.subplots()
# ax.hist(df['ApplicantIncome'], edgecolor="white")
# plt.show()

df['ApplicantIncome_log'] = np.log(df['ApplicantIncome'] + 1)
fig, ax = plt.subplots()
ax.hist(df['ApplicantIncome_log'], edgecolor="white")
plt.show()