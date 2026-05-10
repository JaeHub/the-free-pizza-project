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
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

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


def evaluate_test(y_test: np.ndarray, test_probs: np.ndarray, model_name: str) -> float:
    """Print test AUC, plot ROC curve and confusion matrix at the 0.5 threshold."""
    test_auc = roc_auc_score(y_test, test_probs)
    print(f"[{model_name}] Test AUC: {test_auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, test_probs)
    y_pred = (test_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"{model_name}: ROC curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_title(f"{model_name}: confusion matrix @ 0.5")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    for (i, j), v in np.ndenumerate(cm):
        axes[1].text(j, i, str(v), ha="center", va="center",
                     color="white" if v > cm.max() / 2 else "black")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
    return test_auc


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
