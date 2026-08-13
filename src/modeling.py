"""Reusable baseline modelling and evaluation functions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(
    categorical_columns: list[str], numeric_columns: list[str]
) -> ColumnTransformer:
    """Impute and encode categoricals; impute and standardise numerics."""
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical, categorical_columns),
            ("numeric", numeric, numeric_columns),
        ],
        remainder="drop",
    )


class BalancedMLPClassifier(MLPClassifier):
    """Small neural net. sklearn MLP has no class_weight; oversample the minority class."""

    def fit(self, X, y, sample_weight=None):
        # Do not pass sample_weight: older sklearn MLP.fit() rejects it.
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()
        classes, counts = np.unique(y_arr, return_counts=True)
        max_count = int(counts.max())
        rng = np.random.RandomState(self.random_state)
        parts_x = []
        parts_y = []
        for cls in classes:
            idx = np.flatnonzero(y_arr == cls)
            chosen = rng.choice(idx, size=max_count, replace=len(idx) < max_count)
            parts_x.append(X_arr[chosen])
            parts_y.append(y_arr[chosen])
        order = rng.permutation(max_count * len(classes))
        return super().fit(
            np.concatenate(parts_x, axis=0)[order],
            np.concatenate(parts_y, axis=0)[order],
        )


def make_models(
    categorical_columns: list[str], numeric_columns: list[str]
) -> dict[str, Pipeline]:
    """Return two baseline pipelines with the same ColumnTransformer.

    Logistic Regression is the linear, interpretable baseline. The second model is
    a small neural network so at least one algorithm differs from typical
    Practical Data Science tree / kNN work.
    """
    shared_preprocess = make_preprocessor(categorical_columns, numeric_columns)
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", shared_preprocess),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2_000, class_weight="balanced", random_state=42
                    ),
                ),
            ]
        ),
        "Neural Network": Pipeline(
            steps=[
                (
                    "preprocess",
                    make_preprocessor(categorical_columns, numeric_columns),
                ),
                # One-hot columns are 0/1; scale them too because MLP is scale-sensitive.
                ("scale_all", StandardScaler(with_mean=False)),
                (
                    "model",
                    BalancedMLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        max_iter=400,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=20,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics at a fixed probability threshold."""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "pr_auc_average_precision": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def evaluate_models(
    models: dict[str, Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, list[list[int]]]]:
    """Fit models and evaluate them on the test set.

    Thresholds are optional. When omitted, every model uses 0.5. Ranking metrics
    (PR-AUC, ROC-AUC) do not depend on the threshold.
    """
    thresholds = thresholds or {}
    rows = []
    fitted = {}
    matrices = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        threshold = thresholds.get(name, 0.5)
        metrics = metrics_at_threshold(y_test, probabilities, threshold=threshold)
        predictions = (probabilities >= threshold).astype(int)
        fitted[name] = model
        matrices[name] = confusion_matrix(y_test, predictions).tolist()
        rows.append({"model": name, **metrics})

    return pd.DataFrame(rows), fitted, matrices


def threshold_scan(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate precision, recall, and F1 across candidate thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37)
    rows = [
        metrics_at_threshold(y_true, probabilities, threshold=float(threshold))
        for threshold in thresholds
    ]
    return pd.DataFrame(rows)


def tune_threshold_for_f1(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """Select the threshold that maximises F1 on a validation set only."""
    scan = threshold_scan(y_true, probabilities, thresholds=thresholds)
    best_row = scan.loc[scan["f1"].idxmax()]
    return float(best_row["threshold"]), scan


def get_feature_names(pipeline: Pipeline) -> np.ndarray:
    """Return transformed feature names from a fitted preprocessing pipeline."""
    return pipeline.named_steps["preprocess"].get_feature_names_out()


def extract_model_insights(
    pipeline: Pipeline,
    top_n: int = 15,
    x: pd.DataFrame | None = None,
    y: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Return the strongest model contributions.

    Logistic Regression: signed coefficients (positive = higher predicted risk).
    Neural Network: permutation importance on original columns (PR-AUC drop).
    """
    model = pipeline.named_steps["model"]

    if hasattr(model, "coef_"):
        names = get_feature_names(pipeline)
        clean_names = [
            name.replace("categorical__", "").replace("numeric__", "") for name in names
        ]
        values = model.coef_.ravel()
        frame = pd.DataFrame(
            {
                "feature": clean_names,
                "value": values,
                "abs_value": np.abs(values),
                "interpretation": np.where(
                    values >= 0,
                    "associated with higher risk",
                    "associated with lower risk",
                ),
                "source": "logistic_coefficient",
            }
        )
    else:
        if x is None or y is None:
            raise ValueError(
                "Neural network insights require x and y for permutation importance."
            )
        x_used = x
        y_used = y
        if len(x_used) > 2_500:
            x_used = x_used.sample(n=2_500, random_state=42)
            y_used = y_used.loc[x_used.index]
        result = permutation_importance(
            pipeline,
            x_used,
            y_used,
            n_repeats=8,
            random_state=42,
            scoring="average_precision",
            n_jobs=1,
        )
        frame = pd.DataFrame(
            {
                "feature": list(x_used.columns),
                "value": result.importances_mean,
                "abs_value": np.abs(result.importances_mean),
                "interpretation": "PR-AUC drop when the feature is shuffled",
                "source": "permutation_importance",
            }
        )

    return frame.sort_values("abs_value", ascending=False).head(top_n).reset_index(drop=True)


def plot_top_features(
    insights: pd.DataFrame,
    title: str,
    output_path: Path,
    show: bool = False,
) -> None:
    """Save a horizontal bar chart of the strongest model contributions."""
    plot_frame = insights.sort_values("abs_value", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = (
        ["#c0392b" if value >= 0 else "#1f6f8b" for value in plot_frame["value"]]
        if insights["source"].iloc[0] == "logistic_coefficient"
        else ["#1769aa"] * len(plot_frame)
    )
    ax.barh(plot_frame["feature"], plot_frame["value"], color=colors)
    ax.set_xlabel(
        "Coefficient (standardised / one-hot space)"
        if insights["source"].iloc[0] == "logistic_coefficient"
        else "Permutation importance (mean PR-AUC drop)"
    )
    ax.set_ylabel("")
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def plot_threshold_curves(
    scan: pd.DataFrame,
    title: str,
    output_path: Path,
    selected_threshold: float | None = None,
    show: bool = False,
) -> None:
    """Save precision / recall / F1 against candidate thresholds."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(scan["threshold"], scan["precision"], label="Precision")
    ax.plot(scan["threshold"], scan["recall"], label="Recall")
    ax.plot(scan["threshold"], scan["f1"], label="F1-score")
    if selected_threshold is not None:
        ax.axvline(
            selected_threshold,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Selected threshold={selected_threshold:.2f}",
        )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def save_confusion_matrices(
    dataset_name: str,
    matrices: dict[str, list[list[int]]],
    output_dir: Path,
    suffix: str = "",
    show: bool = False,
) -> None:
    """Save a predicted-vs-actual heatmap for each model's confusion matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, matrix in matrices.items():
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Not at risk", "At risk"],
            yticklabels=["Not at risk", "At risk"],
            ax=ax,
        )
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        title_suffix = f" ({suffix})" if suffix else ""
        ax.set_title(f"{dataset_name}: {model_name}{title_suffix}")
        fig.tight_layout()
        slug = model_name.lower().replace(" ", "_")
        extra = f"_{suffix.lower().replace(' ', '_')}" if suffix else ""
        fig.savefig(
            output_dir / f"{dataset_name.lower()}_{slug}{extra}_confusion.png",
            dpi=180,
        )
        if show:
            plt.show()
        plt.close(fig)
