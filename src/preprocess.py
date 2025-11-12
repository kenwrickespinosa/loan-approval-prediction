import pandas as pd

# Load data
df = pd.read_csv("data/loan_data.csv")
print(df.head())

# Check missing values
# print(df.isnull().sum())

# print(df['Gender'].value_counts())

# Handle Gender column
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0]).astype('int')

print(df.info())