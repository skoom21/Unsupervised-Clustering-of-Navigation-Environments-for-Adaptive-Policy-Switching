import sys
from pathlib import Path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config
from src.preprocessing import (
    prepare_numeric_features,
    handle_missing_values,
    create_all_splits,
)
from src.classification import (
    train_knn,
    train_svm,
    train_decision_tree,
    train_bagging,
    train_boosting,
    evaluate_classifier,
    compare_all_classifiers,
    save_task_metrics,
)

np.random.seed(42)

METADATA_FEATURES = ["corridor_width", "num_rooms", "known_obstacle_pct", "map_resolution"]


def _ensure_output_dirs() -> None:
    for path in [config.OUTPUTS_DIR, config.FIGURES_DIR, config.MODELS_DIR, config.REPORTS_DIR]:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(f"Failed to create directory {path}: {exc}")


def _load_labeled_dataset() -> pd.DataFrame:
    labeled_path = config.PROCESSED_DATA_DIR / "labeled_dataset.csv"
    if not labeled_path.exists():
        raise FileNotFoundError(f"Labeled dataset not found: {labeled_path}")
    try:
        return pd.read_csv(labeled_path)
    except Exception as exc:
        logger.error(f"Failed to read labeled dataset: {exc}")
        raise


def _select_feature_columns(df: pd.DataFrame) -> list:
    removed = [col for col in METADATA_FEATURES if col in df.columns]
    feature_cols = [col for col in df.columns if col not in removed]
    logger.info(f"Excluding metadata features: {removed}")
    logger.info(f"Using {len(feature_cols)} grid-derived features for classification")
    return feature_cols


def _save_phase4_scaler(scaler: StandardScaler) -> None:
    scaler_path = config.MODELS_DIR / "scaler_phase4.pkl"
    try:
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved Phase 4 scaler to {scaler_path}")
    except OSError as exc:
        logger.warning(f"Failed to save Phase 4 scaler: {exc}")


def _check_map_name_duplicates(map_names: pd.Series) -> None:
    dup_count = map_names.duplicated().sum()
    if dup_count > 0:
        logger.warning(f"Found {dup_count} duplicate map_name entries")
    else:
        logger.info("No duplicate map_name entries found")


def _check_train_test_overlap(names_train: pd.Series, names_test: pd.Series) -> None:
    overlap = set(names_train).intersection(set(names_test))
    if overlap:
        logger.warning(f"Train/test overlap detected for {len(overlap)} map(s)")
    else:
        logger.info("No map_name overlap between train/test splits")


def _permutation_sanity_check(X_train, y_train, X_test, y_test) -> None:
    rng = np.random.RandomState(config.RANDOM_STATE)
    y_perm = rng.permutation(np.array(y_train))
    model = DecisionTreeClassifier(random_state=config.RANDOM_STATE)
    model.fit(X_train, y_perm)
    perm_acc = np.mean(model.predict(X_test) == np.array(y_test))
    logger.info(f"[SANITY] Permutation accuracy (should ~0.25): {perm_acc:.4f}")


def _save_feature_importances(model, feature_columns: list) -> None:
    importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
    top = importances.head(10)
    logger.info("Top feature importances (Decision Tree):")
    for name, value in top.items():
        logger.info(f"  {name}: {value:.4f}")
    report_path = config.REPORTS_DIR / "map_type_feature_importances.csv"
    try:
        importances.to_csv(report_path, header=["importance"])
        logger.info(f"Saved feature importances to {report_path}")
    except OSError as exc:
        logger.warning(f"Failed to save feature importances: {exc}")


def run_phase_4() -> None:
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: Classification")
    logger.info("=" * 60)

    _ensure_output_dirs()
    df_labeled = _load_labeled_dataset()

    if "map_name" not in df_labeled.columns:
        raise ValueError("map_name column is required for leakage checks")
    map_names = df_labeled["map_name"]
    _check_map_name_duplicates(map_names)

    X_numeric, labels_dict = prepare_numeric_features(df_labeled)
    X_clean = handle_missing_values(X_numeric)

    feature_columns = _select_feature_columns(X_clean)
    X_selected = X_clean[feature_columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    _save_phase4_scaler(scaler)

    splits = create_all_splits(X_scaled, labels_dict)

    primary_task = "label_map_type"
    if primary_task not in splits:
        logger.warning(f"Primary task {primary_task} not found. Skipping.")
    else:
        X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
            X_scaled,
            labels_dict[primary_task],
            map_names,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=labels_dict[primary_task],
        )
        _check_train_test_overlap(pd.Series(names_train), pd.Series(names_test))
        _permutation_sanity_check(X_train, y_train, X_test, y_test)
        results = {}

        knn = train_knn(X_train, y_train, task_name="map_type")
        results["knn"] = evaluate_classifier(knn, X_test, y_test, "knn_map_type")

        svm = train_svm(X_train, y_train, task_name="map_type")
        results["svm"] = evaluate_classifier(svm, X_test, y_test, "svm_map_type")

        dt = train_decision_tree(X_train, y_train, task_name="map_type")
        results["dt"] = evaluate_classifier(dt, X_test, y_test, "dt_map_type")
        _save_feature_importances(dt, feature_columns)

        bagging = train_bagging(X_train, y_train, task_name="map_type")
        results["bagging"] = evaluate_classifier(bagging, X_test, y_test, "bagging_map_type")

        boosting = train_boosting(X_train, y_train, task_name="map_type")
        results["boosting"] = evaluate_classifier(boosting, X_test, y_test, "boosting_map_type")

        compare_all_classifiers(results)
        save_task_metrics(results, "classification_map_type_metrics.csv")

    secondary_tasks = ["label_difficulty", "label_density_category"]
    for task in secondary_tasks:
        if task not in splits:
            logger.warning(f"Secondary task {task} not found. Skipping.")
            continue

        X_train, X_test, y_train, y_test = splits[task]
        task_short = task.replace("label_", "")
        task_results = {}

        knn = train_knn(X_train, y_train, task_name=task_short)
        task_results["knn"] = evaluate_classifier(knn, X_test, y_test, f"knn_{task_short}")

        svm = train_svm(X_train, y_train, task_name=task_short)
        task_results["svm"] = evaluate_classifier(svm, X_test, y_test, f"svm_{task_short}")

        dt = train_decision_tree(X_train, y_train, task_name=task_short)
        task_results["dt"] = evaluate_classifier(dt, X_test, y_test, f"dt_{task_short}")

        save_task_metrics(task_results, f"classification_{task_short}_metrics.csv")

    logger.info("Phase 4 classification complete")


def main() -> None:
    try:
        run_phase_4()
    except Exception as exc:
        logger.error(f"Phase 4 failed: {exc}")


if __name__ == "__main__":
    main()
