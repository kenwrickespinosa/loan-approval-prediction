from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data
from features_selection import FeaturesSelection
from imblearn.over_sampling import SMOTE
import joblib

models = {
    "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42),
    "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
    )
}

df = preprocess_data("data/loan_data.csv")

fs = FeaturesSelection(df)
X, y = fs.get_original_features()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smt = SMOTE(random_state=42)
X_train_smt, y_train_smt = smt.fit_resample(X_train, y_train)

# Debugging: check class distribution
print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_smt.value_counts())

# grandient_boosting = GradientBoostingClassifier(random_state=42)
# grandient_boosting.fit(X_train, y_train)
# y_pred = grandient_boosting.predict(X_test)
# acc = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# print(f"Gradient Boosting: Accuracy={acc:.4f}, F1={f1:.4f}")

for name, model in models.items():
    model.fit(X_train_smt, y_train_smt)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f}")
    print(confusion_matrix(y_test, y_pred))