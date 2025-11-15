class FeatureSelection:
    def __init__(self, df):
        self.df = df.copy()
    
    def get_original_features(self):
        X = self.df[['Gender', 'Married', 'Dependents', 'Education',
                'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                'Loan_Amount_Term', 'Credit_History', 'Property_Area']]
        y = self.df['Loan_Status']
        return X, y
    
    def get_scaled_features(self):
        X = self.df[['Gender', 'Married', 'Dependents', 'Education',
                'Self_Employed', 'Credit_History', 'Property_Area', 'ApplicantIncome_scale',
                'CoapplicantIncome_cap_scale', 'LoanAmount_scale', 'LoanAmountTerm_scale']]
        y = self.df['Loan_Status']
        return X, y
