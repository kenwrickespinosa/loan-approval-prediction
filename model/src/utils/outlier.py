def define_outlier(df, col):
    q1 = df[col].quantile(.25)
    q3 = df[col].quantile(.75)
    iqr = q3 - q1
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)
    return lower, upper

def cap_outlier(df, col):
    lower, upper = define_outlier(df, col)
    return df[col].clip(lower, upper)