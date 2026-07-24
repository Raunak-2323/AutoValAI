# train.py

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("car data.csv")

# ==========================
# FEATURE ENGINEERING
# ==========================

current_year = 2025
df["Car_Age"] = current_year - df["Year"]
df.drop("Year", axis=1, inplace=True)

# ==========================
# INPUTS AND TARGET
# ==========================

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# ==========================
# NUMERICAL & CATEGORICAL
# ==========================

num_cols = X.select_dtypes(exclude="object").columns
cat_cols = X.select_dtypes(include="object").columns

# ==========================
# PREPROCESSING
# ==========================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# ==========================
# MODEL PIPELINE
# ==========================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# HYPERPARAMETER TUNING
# ==========================

params = {
    "model__n_estimators": [100, 200, 300, 500],
    "model__max_depth": [5, 10, 15, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"]
}

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=params,
    n_iter=20,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1
)

# ==========================
# TRAIN MODEL
# ==========================

print("Training started...\n")

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("Best Parameters:")
print(search.best_params_)

# ==========================
# EVALUATION
# ==========================

y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("\nModel Performance")
print("-" * 30)
print(f"R2 Score : {r2:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")

# ==========================
# SAVE MODEL
# ==========================

with open("car_price_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\nModel saved as car_price_model.pkl")
