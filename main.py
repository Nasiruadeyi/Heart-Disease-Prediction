# main.py
"""
Main script to run the full Heart Disease Prediction pipeline:
1. Load data
2. Preprocess
3. Train and compare baseline models
4. Hyperparameter tuning for Random Forest
5. Final evaluation and feature importance
"""

from src.data_loader import load_data, preprocess_data
from src.model_training import train_models, tune_random_forest
from src.evaluation import evaluate_model

def main():
    print("Loading dataset...")
    df = load_data()  # default path: data/heart_stat.xlsx
    print("Dataset loaded. Shape:", df.shape)

    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("Preprocessing done.")
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    print("\nTraining baseline models...")
    models = train_models(X_train, y_train, X_test, y_test)

    print("\nHyperparameter tuning for Random Forest...")
    best_rf = tune_random_forest(X_train, y_train)

    print("\nEvaluating final tuned Random Forest model...")
    evaluate_model(best_rf, X_test, y_test)

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
