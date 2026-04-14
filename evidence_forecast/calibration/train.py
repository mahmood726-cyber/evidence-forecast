"""Flip-forecaster training: GBM (shipped), RF and L1-LR (comparators).

Temporal split + cardiology holdout per spec §5.4.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

_NON_FEATURE_COLS = {"ma_id", "flip", "topic_area", "v1_date"}


@dataclass(frozen=True)
class TrainingArtifacts:
    feature_names: list[str]
    gbm_path: Path
    rf_path: Path
    l1_path: Path


def split_temporal(
    df: pd.DataFrame, cutoff: str, holdout_topic: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["v1_date"] = pd.to_datetime(df["v1_date"])
    cut = pd.Timestamp(cutoff)
    train = df[df["v1_date"] < cut].copy()
    test = df[df["v1_date"] >= cut].copy()
    if holdout_topic:
        train = train[train["topic_area"] != holdout_topic].copy()
    return train, test


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


def _make_pipeline(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", model),
    ])


def train_models(
    df: pd.DataFrame,
    models_dir: Path,
    cutoff: str = "2023-01-01",
    holdout_topic: str | None = "cardiology",
    seed: int = 0,
) -> TrainingArtifacts:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    train, _ = split_temporal(df, cutoff=cutoff, holdout_topic=holdout_topic)
    features = _feature_cols(train)
    X = train[features].apply(pd.to_numeric, errors="coerce").values
    y = train["flip"].astype(int).values

    if XGBClassifier is not None:
        gbm = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, eval_metric="logloss", tree_method="hist",
        )
    else:
        gbm = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=seed,
        )
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=5,
        n_jobs=-1, random_state=seed,
    )
    l1 = LogisticRegression(
        penalty="l1", solver="liblinear", C=0.5,
        max_iter=2000, random_state=seed,
    )

    gbm_pipe = _make_pipeline(gbm)
    rf_pipe = _make_pipeline(rf)
    l1_pipe = _make_pipeline(l1)

    gbm_pipe.fit(X, y)
    rf_pipe.fit(X, y)
    l1_pipe.fit(X, y)

    gbm_path = models_dir / "flip_forecaster_v1.pkl"
    rf_path = models_dir / "flip_forecaster_v1_rf.pkl"
    l1_path = models_dir / "flip_forecaster_v1_l1.pkl"
    _persist(gbm_pipe, gbm_path, features)
    _persist(rf_pipe, rf_path, features)
    _persist(l1_pipe, l1_path, features)
    return TrainingArtifacts(features, gbm_path, rf_path, l1_path)


def _persist(pipeline: Pipeline, path: Path, features: list[str]) -> None:
    with open(path, "wb") as f:
        pickle.dump({"pipeline": pipeline, "features": features, "schema_version": "1.0.0"}, f)
