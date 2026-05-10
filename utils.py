"""Shared infrastructure for the pizza-request modeling notebooks.

Centralizes data loading, the train/test split, 5-fold stratified CV folds,
feature engineering, prediction artifact I/O, and evaluation helpers so
every model uses byte-identical splits and engineered features.
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

CLASS_LABELS = ["Not fulfilled", "Fulfilled"]
LR_BASELINE_AUC = 0.6452

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

REPO_ROOT = Path(__file__).resolve().parent
SPLITS_DIR = REPO_ROOT / "splits"
PREDICTIONS_DIR = REPO_ROOT / "predictions"
RESULTS_CSV = REPO_ROOT / "results.csv"

TARGET_COLUMN = "requester_received_pizza"

# Numeric/boolean fields from CLAUDE.md that exist in both train and test.
# `post_was_edited` is intentionally omitted: it appears in train but not in
# the Kaggle test split (verified 2026-05), and CLAUDE.md flags it as
# "if present in test".
FEATURE_COLUMNS = [
    "requester_account_age_in_days_at_request",
    "requester_days_since_first_post_on_raop_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "unix_timestamp_of_request",
]


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """Download (cached) the Kaggle pizza dataset and return train.json as a DataFrame."""
    import kagglehub

    path = kagglehub.dataset_download("kaggle/random-acts-of-pizza")
    return pd.read_json(os.path.join(path, "train.json"))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the modeling feature matrix from the raw DataFrame.

    Selects the leakage-safe numeric features, derives request-hour and
    day-of-week from the unix timestamp, and adds simple text-derived
    counts/flags from title and body. Returns a features-only DataFrame
    aligned to ``df.index``.
    """
    out = df[FEATURE_COLUMNS].copy()

    ts = pd.to_datetime(out["unix_timestamp_of_request"], unit="s", utc=True)
    out["request_hour"] = ts.dt.hour.astype(int)
    out["request_dow"] = ts.dt.dayofweek.astype(int)
    out = out.drop(columns=["unix_timestamp_of_request"])

    title = df["request_title"].fillna("").astype(str)
    body = df["request_text_edit_aware"].fillna("").astype(str)

    out["title_word_count"] = title.str.split().str.len().astype(int)
    out["title_char_count"] = title.str.len().astype(int)
    out["body_word_count"] = body.str.split().str.len().astype(int)
    out["body_char_count"] = body.str.len().astype(int)
    out["has_url"] = body.str.contains("http", case=False, regex=False).astype(int)
    out["has_question_mark"] = body.str.contains("?", regex=False).astype(int)
    out["has_exclamation"] = body.str.contains("!", regex=False).astype(int)

    return out


def get_train_test_split(
    y: np.ndarray, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray]:
    """Return cached stratified (train_idx, test_idx) over ``range(len(y))``.

    Persists to ``splits/train_idx.npy`` and ``splits/test_idx.npy`` so every
    notebook sees the same held-out set. Validates a cached split against the
    current ``len(y)`` before reusing it.
    """
    SPLITS_DIR.mkdir(exist_ok=True)
    train_path = SPLITS_DIR / "train_idx.npy"
    test_path = SPLITS_DIR / "test_idx.npy"

    if train_path.exists() and test_path.exists():
        train_idx = np.load(train_path)
        test_idx = np.load(test_path)
        if len(train_idx) + len(test_idx) == len(y):
            return train_idx, test_idx

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=y, random_state=random_state
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    np.save(train_path, train_idx)
    np.save(test_path, test_idx)
    return train_idx, test_idx


def get_cv_folds(
    y_train: np.ndarray, n_splits: int = 5, random_state: int = RANDOM_STATE
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return cached stratified k-fold splits as a list of (train_fold, val_fold) index pairs.

    Indices are positional within ``y_train`` (i.e. range(len(y_train))).
    """
    SPLITS_DIR.mkdir(exist_ok=True)
    cache_path = SPLITS_DIR / f"cv_folds_k{n_splits}.npz"

    if cache_path.exists():
        data = np.load(cache_path)
        if data["n_samples"].item() == len(y_train):
            return [
                (data[f"train_{i}"], data[f"val_{i}"]) for i in range(n_splits)
            ]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = [(tr, va) for tr, va in skf.split(np.zeros(len(y_train)), y_train)]

    payload = {"n_samples": np.array(len(y_train))}
    for i, (tr, va) in enumerate(folds):
        payload[f"train_{i}"] = tr
        payload[f"val_{i}"] = va
    np.savez(cache_path, **payload)
    return folds


def save_predictions(model_name: str, oof_probs: np.ndarray, test_probs: np.ndarray) -> None:
    """Persist OOF and test probabilities for cross-model comparison."""
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    np.save(PREDICTIONS_DIR / f"{model_name}_oof.npy", oof_probs)
    np.save(PREDICTIONS_DIR / f"{model_name}_test.npy", test_probs)


def evaluate_cv(y_train: np.ndarray, oof_probs: np.ndarray, model_name: str) -> dict:
    """Compute per-fold and aggregate AUC from out-of-fold predictions."""
    folds = get_cv_folds(y_train)
    fold_aucs = [
        roc_auc_score(y_train[val_idx], oof_probs[val_idx]) for _, val_idx in folds
    ]
    mean = float(np.mean(fold_aucs))
    std = float(np.std(fold_aucs))
    overall = roc_auc_score(y_train, oof_probs)
    print(
        f"[{model_name}] CV AUC: {mean:.4f} ± {std:.4f}  "
        f"(per-fold: {[round(a, 4) for a in fold_aucs]}, overall OOF: {overall:.4f})"
    )
    return {"mean": mean, "std": std, "fold_aucs": fold_aucs, "overall_oof": overall}


def describe_target(y: np.ndarray, df: pd.DataFrame | None = None) -> None:
    """Print target value counts and class imbalance ratio in CNN-notebook format."""
    if df is not None and TARGET_COLUMN in df.columns:
        print(df[TARGET_COLUMN].value_counts())
        ratio = df[TARGET_COLUMN].value_counts(normalize=True).round(3).to_dict()
    else:
        s = pd.Series(y).map({1: True, 0: False}).rename(TARGET_COLUMN)
        print(s.value_counts())
        ratio = s.value_counts(normalize=True).round(3).to_dict()
    print(f"\nClass imbalance ratio: {ratio}")


def evaluate_test(
    y_test: np.ndarray, test_probs: np.ndarray, model_name: str, threshold: float = 0.5
) -> dict:
    """Print test AUC, classification report, confusion-matrix heatmap, and ROC curve.

    Returns a dict of summary metrics so the notebook's final summary cell can
    reuse them without recomputing.
    """
    y_pred = (test_probs >= threshold).astype(int)
    test_auc = roc_auc_score(y_test, test_probs)
    f1_fulfilled = f1_score(y_test, y_pred)
    accuracy = float((y_pred == y_test).mean())

    print(f"Test ROC-AUC: {test_auc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_LABELS))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, test_probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {test_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {"test_auc": test_auc, "f1_fulfilled": f1_fulfilled, "accuracy": accuracy}


def plot_cv_fold_aucs(fold_aucs, mean: float, model_name: str) -> None:
    """Bar chart of per-fold validation AUC with a dashed line at the mean."""
    fold_aucs = list(fold_aucs)
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(fold_aucs) + 1), fold_aucs, color="steelblue")
    plt.axhline(mean, linestyle="--", color="black", linewidth=1, label=f"mean = {mean:.4f}")
    plt.ylim(min(fold_aucs) - 0.02, max(fold_aucs) + 0.02)
    plt.xticks(range(1, len(fold_aucs) + 1))
    plt.xlabel("Fold")
    plt.ylabel("Validation AUC")
    plt.title(f"Per-fold CV AUC — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cv_learning_curves(
    evals_results_per_fold, model_name: str, metric_key: str = "auc"
) -> None:
    """Plot per-fold val-AUC vs n_trees curves on a single axis.

    Accepts each element as either an XGBoost-style dict (``{"validation_0": {"auc": [...]}}``)
    or a LightGBM-style dict (``{"valid_0": {"auc": [...]}}``); the first inner
    key is used automatically.
    """
    plt.figure(figsize=(15, 4))
    for i, evals in enumerate(evals_results_per_fold):
        outer_key = next(iter(evals))
        curve = evals[outer_key][metric_key]
        plt.plot(curve, label=f"fold {i}")
    plt.xlabel("Trees")
    plt.ylabel(f"Validation {metric_key.upper()}")
    plt.title(f"CV learning curves — {model_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def inspect_predictions(
    text_series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    n: int = 3,
) -> None:
    """Show top-N high-confidence correct and confident-wrong predictions as DataFrames.

    Mirrors CNN Cell 21 layout. ``text_series`` is whatever readable text identifies
    each test sample (e.g. ``df.iloc[test_idx]["request_title"]``).
    """
    try:
        from IPython.display import display
    except ImportError:
        display = print

    text = pd.Series(text_series).reset_index(drop=True)
    confidence = np.where(y_pred == 1, y_prob, 1 - y_prob)
    df = pd.DataFrame({
        "text": text,
        "actual": y_true,
        "predicted": y_pred,
        "confidence": confidence,
    })
    correct = df[df["actual"] == df["predicted"]]
    wrong = df[df["actual"] != df["predicted"]]

    print(f"=== {model_name}: Correct high-confidence predictions ===")
    display(correct.sort_values("confidence", ascending=False).head(n)[["text", "actual", "confidence"]])
    print(f"\n=== {model_name}: Confident wrong predictions ===")
    display(wrong.sort_values("confidence", ascending=False).head(n)[["text", "actual", "predicted", "confidence"]])


def print_summary(
    model_name: str,
    dataset: dict,
    model: dict,
    training: dict,
    test: dict,
    takeaways: list,
) -> None:
    """Emoji-decorated summary block matching cnn_model.ipynb Cell 24's layout."""
    bar = "=" * 55
    print(bar)
    print(f"       {model_name.upper()} — KEY FINDINGS SUMMARY")
    print(bar)

    print(f"\n📦 Dataset")
    print(f"   Total samples      : {dataset['total_samples']:,}")
    print(f"   Fulfillment rate   : {dataset['fulfillment_rate_pct']:.1f}%  (class imbalance 75/25)")
    print(f"   Features used      : {dataset['n_features']}  (leakage-safe)")

    print(f"\n🧠 Model — {model['name']}")
    print(f"   Hyperparameters    : {model['params_summary']}")
    print(f"   Class handling     : {model['class_handling']}")

    print(f"\n📈 Cross-validation")
    print(f"   CV AUC             : {training['cv_auc_mean']:.4f} ± {training['cv_auc_std']:.4f}")
    print(f"   Per-fold AUCs      : {[round(a, 4) for a in training['fold_aucs']]}")
    print(f"   Median best_iter   : {training['median_best_iter']}")
    print(f"   Refit train AUC    : {training['refit_train_auc']:.4f}  "
          f"(gap vs CV = {training['refit_train_auc'] - training['cv_auc_mean']:+.4f})")

    print(f"\n🎯 Test Performance")
    baseline = test.get("lr_baseline", LR_BASELINE_AUC)
    delta = test["roc_auc"] - baseline
    print(f"   ROC-AUC            : {test['roc_auc']:.4f}  (LR baseline = {baseline:.4f}, Δ = {delta:+.4f})")
    print(f"   F1 (fulfilled)     : {test['f1_fulfilled']:.4f}")
    print(f"   Accuracy           : {test['accuracy_pct']:.1f}%")

    print(f"\n💡 Key Takeaways")
    for line in takeaways:
        print(f"   • {line}")
    print(bar)


def plot_feature_importance(
    importances: np.ndarray, feature_names, model_name: str, top_n: int = 20
) -> None:
    """Horizontal bar plot of the top-N feature importances."""
    importances = np.asarray(importances)
    feature_names = np.asarray(list(feature_names))
    order = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(8, max(3, 0.3 * len(order))))
    plt.barh(range(len(order)), importances[order])
    plt.yticks(range(len(order)), feature_names[order])
    plt.xlabel("Importance")
    plt.title(f"{model_name}: top {len(order)} features")
    plt.tight_layout()
    plt.show()


def log_run(
    model_name: str,
    params: dict,
    cv_mean: float,
    cv_std: float,
    test_auc: float,
    results_path: Path = RESULTS_CSV,
) -> None:
    """Append a single-row run record to ``results.csv``."""
    row = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "model": model_name,
        "cv_auc_mean": round(cv_mean, 4),
        "cv_auc_std": round(cv_std, 4),
        "test_auc": round(test_auc, 4),
        "params": str(params),
    }
    df = pd.DataFrame([row])
    df.to_csv(
        results_path,
        mode="a",
        header=not Path(results_path).exists(),
        index=False,
    )
