# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path="data/heart_stat.xlsx"):
    """
    Load the raw Excel dataset.
    Default path points to 'data/heart_stat.xlsx'.
    """
    df = pd.read_excel(file_path)
    return df

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Split the dataset into training and test sets and scale numerical features.
    Returns X_train_scaled, X_test_scaled, y_train, y_test
    """
    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale numerical columns
    num_cols = [
        "age",
        "resting bp s",
        "cholesterol",
        "max heart rate",
        "oldpeak"
    ]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    return X_train_scaled, X_test_scaled, y_train, y_test
