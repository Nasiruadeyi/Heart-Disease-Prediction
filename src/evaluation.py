# src/evaluation.py
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(model, X_test, y_test, plot_cm=True, plot_importance=True):
    """
    Evaluate a trained model: metrics, confusion matrix, and feature importance.
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("Final Model Results")
    print("Accuracy:", accuracy)
    print("ROC-AUC:", roc_auc)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    if plot_cm:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[0, 1],
                    yticklabels=[0, 1])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    # Feature Importance (for tree-based models)
    if plot_importance and hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame({
            "feature": X_test.columns,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        display(feature_importance)

        plt.figure(figsize=(8, 6))
        sns.barplot(
            data=feature_importance,
            x="importance",
            y="feature"
        )
        plt.title("Feature Importance")
        plt.show()
