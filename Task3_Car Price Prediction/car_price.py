# Car price prediction 
# Written by: Arshath Abdulla A

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings

warnings.filterwarnings("ignore")
DATA_PATH = r"C:\Users\ARSHATH ABDULLA A\OneDrive\Desktop\Internships\OASIS\OIBSIP\Task3 Car price prediction\car data.csv"
def load_data(path):
    df = pd.read_csv(path)
    print("Loaded shape:", df.shape)
    return df

def basic_cleanup(df):
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates()
    if "Selling_Price" not in df.columns:
        for alt in ["SellingPrice", "Price", "sell_price", "Selling Price"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "Selling_Price"})
                break
    return df

def prepare_features(df):
    if "Selling_Price" not in df.columns:
        raise ValueError("Selling_Price column not found in dataset.")
    y = df["Selling_Price"].astype(float)
    keep = []
    for c in ["Year", "Present_Price", "Driven_kms", "Fuel_Type", "Selling_type", "Transmission", "Owner"]:
        if c in df.columns:
            keep.append(c)
    X = df[keep].copy()
    if "Year" in X.columns:
        current_year = pd.Timestamp.now().year
        X["Car_Age"] = current_year - X["Year"]
        X = X.drop(columns=["Year"])
    if "Driven_kms" in X.columns:
        X["Driven_kms"] = pd.to_numeric(X["Driven_kms"], errors="coerce").fillna(0)
    if "Present_Price" in X.columns:
        X["Present_Price"] = pd.to_numeric(X["Present_Price"], errors="coerce").fillna(0)

    cat_cols = [c for c in ["Fuel_Type", "Selling_type", "Transmission", "Owner"] if c in X.columns]
    ohe_kwargs = {}
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformer = ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_cols)
        ],
        remainder="passthrough"
    )

    X_trans = transformer.fit_transform(X)
    feature_names = None
    try:
        cat_feature_names = []
        if len(cat_cols) > 0:
            try:
                cat_feature_names = transformer.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
            except Exception:
                cat_feature_names = transformer.named_transformers_["cat"].get_feature_names(cat_cols).tolist()
        passthrough_cols = [c for c in X.columns if c not in cat_cols]
        feature_names = cat_feature_names + passthrough_cols
        X_df = pd.DataFrame(X_trans, columns=feature_names)
    except Exception:
        X_df = pd.DataFrame(X_trans)
    return X_df, y, transformer

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X.values if hasattr(X, "values") else X,
                                                        y.values, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
    r2_lr = r2_score(y_test, pred_lr)
    print("LinearRegression RMSE: {:.3f}  R2: {:.3f}".format(rmse_lr, r2_lr))
    rf = RandomForestRegressor(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
    r2_rf = r2_score(y_test, pred_rf)
    print("RandomForest RMSE: {:.3f}  R2: {:.3f}".format(rmse_rf, r2_rf))
    if rmse_rf <= rmse_lr:
        chosen = rf
        print("Selected model: RandomForestRegressor")
        chosen_pred = pred_rf
    else:
        chosen = lr
        print("Selected model: LinearRegression")
        chosen_pred = pred_lr
    print("\nSample predictions (first 8 rows):")
    sample_actual = y_test[:8]
    sample_pred = chosen_pred[:8]
    for a, p in zip(sample_actual, sample_pred):
        print(f"actual: {a:.2f}  predicted: {p:.2f}")
    return chosen, transformer 
def save_model(model, transformer):
    joblib.dump(model, "car_price_model.joblib")
    joblib.dump(transformer, "car_price_transformer.joblib")
    print("\nSaved model -> car_price_model.joblib")
    print("Saved transformer -> car_price_transformer.joblib")
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df = basic_cleanup(df)
    X, y, transformer = prepare_features(df)
    best_model, _ = train_and_evaluate(X, y)
    save_model(best_model, transformer)
