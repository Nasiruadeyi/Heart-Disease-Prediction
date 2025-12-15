# src/model_training.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_models(X_train, y_train, X_test, y_test):
    """
    Train multiple baseline models and return the fitted models dictionary.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    import pandas as pd

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM (RBF)": SVC(probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        results[name] = {
            "model": model,
            "Accuracy": acc,
            "ROC_AUC": roc
        }

        print(f"\n{name}")
        print("Accuracy:", acc)
        print("ROC-AUC:", roc)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    # Summary table
    results_df = pd.DataFrame({k: {"Accuracy": v["Accuracy"], "ROC_AUC": v["ROC_AUC"]} for k, v in results.items()}).T
    print("\nModel Comparison:")
    display(results_df.sort_values("ROC_AUC", ascending=False))

    # Return dictionary of fitted models
    return {k: v["model"] for k, v in results.items()}

def tune_random_forest(X_train, y_train):
    """
    Hyperparameter tuning for Random Forest using GridSearchCV.
    Returns the best estimator.
    """
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("Best ROC-AUC:", grid_search.best_score_)
    print("Best Parameters:", grid_search.best_params_)

    return grid_search.best_estimator_
