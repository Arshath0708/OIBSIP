import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_data(path):
    return pd.read_csv(path)

def prepare_data(df):
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 3))

    print("\nSample Predictions:")
    for a, p in list(zip(y_test.values, preds))[:5]:
        print("Actual:", round(a, 2), "Predicted:", round(p, 2))

    return model

def main():
    path = r"C:\Users\ARSHATH ABDULLA A\OneDrive\Desktop\Internships\OASIS\OIBSIP\Task4_Sales\Advertising.csv"
    df = load_data(path)
    print("Dataset shape:", df.shape)

    X, y = prepare_data(df)
    model = train_model(X, y)

    joblib.dump(model, "sales_prediction_model.joblib")
    print("\nModel saved successfully")

if __name__ == "__main__":
    main()
