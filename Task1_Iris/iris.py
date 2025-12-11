import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
def load_data():
    df = pd.read_csv(r"C:\Users\ARSHATH ABDULLA A\OneDrive\Desktop\Internships\OASIS\OIBSIP\Task1_Iris\Iris.csv")
    X = df.drop("Species", axis=1)
    y = df["Species"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    target_names = le.classes_
    return X, y_encoded, target_names, le
def quick_eda(X, y, target_names):
    print("Shape:", X.shape)
    print("First rows:\n", X.head())
    print("Class counts:")
    for i, name in enumerate(target_names):
        print(f"{name}: {(y == i).sum()}")
def train_and_evaluate(X, y, target_names, label_encoder):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("CV mean accuracy:", cv_scores.mean())
    joblib.dump((model, label_encoder), "iris_rf_model.joblib")
    print("Saved model to iris_rf_model.joblib")
    return model, X_test, y_test, y_pred
def plot_feature_importance(model, X):
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    fi.plot(kind='barh')
    plt.title("Feature importances — Iris RandomForest")
    plt.tight_layout()
    plt.savefig("iris_feature_importance.png")
    print("Saved feature importance plot as iris_feature_importance.png")
if __name__ == "__main__":
    X, y, names, le = load_data()
    quick_eda(X, y, names)
    model, X_test, y_test, y_pred = train_and_evaluate(X, y, names, le)
    plot_feature_importance(model, X)
    species_pred = le.inverse_transform(y_pred)
    print("\nPredicted species names for test set:\n", species_pred[:10])
