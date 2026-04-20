from typing import Dict

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    ConfusionMatrixDisplay,
)

import config

np.random.seed(42)


def _save_model(model, filename: str) -> None:
    path = config.MODELS_DIR / filename
    try:
        joblib.dump(model, path)
        print(f"[OK] Saved model to {path}")
    except OSError as exc:
        print(f"[WARNING] Failed to save model {filename}: {exc}")


def _save_dataframe(df: pd.DataFrame, filename: str) -> None:
    path = config.REPORTS_DIR / filename
    try:
        df.to_csv(path, index=True)
        print(f"[OK] Saved report to {path}")
    except OSError as exc:
        print(f"[WARNING] Failed to save report {filename}: {exc}")


def train_knn(X_train, y_train, task_name: str = "map_type"):
    print(f"=== Training KNN ({task_name}) ===")
    params = {"n_neighbors": [3, 5, 7, 9, 11]}
    model = KNeighborsClassifier()
    grid = GridSearchCV(model, params, cv=5, n_jobs=-1)
    try:
        grid.fit(X_train, y_train)
    except Exception as exc:
        print(f"[ERROR] KNN training failed: {exc}")
        raise

    best = grid.best_estimator_
    _save_model(best, f"knn_{task_name}.pkl")
    return best


def train_svm(X_train, y_train, task_name: str = "map_type"):
    print(f"=== Training SVM ({task_name}) ===")
    model = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    params = {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]}
    grid = GridSearchCV(model, params, cv=5, n_jobs=-1)
    try:
        grid.fit(X_train, y_train)
    except Exception as exc:
        print(f"[ERROR] SVM training failed: {exc}")
        raise

    best = grid.best_estimator_
    _save_model(best, f"svm_{task_name}.pkl")
    return best


def train_decision_tree(X_train, y_train, task_name: str = "map_type"):
    print(f"=== Training Decision Tree ({task_name}) ===")
    model = DecisionTreeClassifier(class_weight="balanced", random_state=42)
    params = {"max_depth": [3, 5, 7, None]}
    grid = GridSearchCV(model, params, cv=5, n_jobs=-1)
    try:
        grid.fit(X_train, y_train)
    except Exception as exc:
        print(f"[ERROR] Decision Tree training failed: {exc}")
        raise

    best = grid.best_estimator_
    _save_model(best, f"dt_{task_name}.pkl")
    return best


def train_bagging(X_train, y_train, task_name: str = "map_type"):
    print(f"=== Training Bagging ({task_name}) ===")
    base = DecisionTreeClassifier(random_state=42)
    model = BaggingClassifier(estimator=base, n_estimators=50, random_state=42)
    try:
        model.fit(X_train, y_train)
    except Exception as exc:
        print(f"[ERROR] Bagging training failed: {exc}")
        raise

    _save_model(model, f"bagging_{task_name}.pkl")
    return model


def train_boosting(X_train, y_train, task_name: str = "map_type"):
    print(f"=== Training Boosting ({task_name}) ===")
    model = AdaBoostClassifier(n_estimators=100, random_state=42)
    try:
        model.fit(X_train, y_train)
    except Exception as exc:
        print(f"[ERROR] Boosting training failed: {exc}")
        raise

    _save_model(model, f"boosting_{task_name}.pkl")
    return model


def evaluate_classifier(model, X_test, y_test, model_name: str) -> Dict[str, float]:
    print(f"=== Evaluating {model_name} ===")
    try:
        y_pred = model.predict(X_test)
    except Exception as exc:
        print(f"[ERROR] Evaluation failed for {model_name}: {exc}")
        raise

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    print(classification_report(y_test, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix: {model_name}")
    fig.tight_layout()
    fig_path = config.FIGURES_DIR / f"cm_{model_name}.png"
    try:
        fig.savefig(fig_path, dpi=200)
        print(f"[OK] Saved confusion matrix to {fig_path}")
    except OSError as exc:
        print(f"[WARNING] Failed to save confusion matrix: {exc}")
    finally:
        plt.close(fig)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compare_all_classifiers(results_dict: Dict[str, Dict[str, float]]) -> None:
    print("=== Comparing classifiers ===")
    df = pd.DataFrame.from_dict(results_dict, orient="index")
    df.index.name = "model"
    _save_dataframe(df, "classifier_comparison.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    df[["accuracy", "precision", "recall", "f1"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Classifier Comparison")
    fig.tight_layout()
    fig_path = config.FIGURES_DIR / "classifier_comparison.png"
    try:
        fig.savefig(fig_path, dpi=200)
        print(f"[OK] Saved classifier comparison plot to {fig_path}")
    except OSError as exc:
        print(f"[WARNING] Failed to save comparison plot: {exc}")
    finally:
        plt.close(fig)


def save_task_metrics(results_dict: Dict[str, Dict[str, float]], filename: str) -> None:
    df = pd.DataFrame.from_dict(results_dict, orient="index")
    df.index.name = "model"
    _save_dataframe(df, filename)
