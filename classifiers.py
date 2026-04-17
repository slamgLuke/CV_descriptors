"""Model training and evaluation helpers for leaf disease classification."""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def train_and_evaluate(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Dict[str, dict]:
    """Train multiple classifiers with 5-fold CV and evaluate metrics.

    Input: Feature matrix X and label vector y.
    Output: Dictionary of model metrics and confusion matrix figures.
    Logic: Run cross-validated predictions for each model and compute metrics.
    """
    if X is None or y is None:
        raise ValueError("X and y must not be None.")

    if len(X) == 0 or len(y) == 0:
        raise ValueError("X and y must not be empty.")

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")

    cross_validator = KFold(n_splits=5, shuffle=True, random_state=random_state)

    classifiers = {
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
        ),
    }

    unique_labels = np.unique(y)
    results: Dict[str, dict] = {}

    for model_name, model in classifiers.items():
        predicted_labels = cross_val_predict(model, X, y, cv=cross_validator, n_jobs=-1)

        accuracy_value = accuracy_score(y, predicted_labels)
        f1_value = f1_score(y, predicted_labels, average="weighted")
        confusion = confusion_matrix(y, predicted_labels, labels=unique_labels)

        figure, axis = plt.subplots(figsize=(6, 5))
        display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=unique_labels)
        display.plot(ax=axis, cmap="Blues", colorbar=False)
        axis.set_title(f"Confusion Matrix - {model_name}")
        figure.tight_layout()

        results[model_name] = {
            "accuracy": float(accuracy_value),
            "f1_score": float(f1_value),
            "confusion_matrix": confusion,
            "confusion_matrix_figure": figure,
        }

    return results
