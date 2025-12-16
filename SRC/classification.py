from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


@dataclass(frozen=True)
class ClassifierConfig:
    max_features: int = 90000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    test_size: float = 0.2
    random_state: int = 42


def make_lr_pipeline(cfg: ClassifierConfig = ClassifierConfig()) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                max_features=cfg.max_features,
                ngram_range=cfg.ngram_range,
                min_df=cfg.min_df,
            )),
            ("model", LogisticRegression(max_iter=3000)),
        ]
    )


def make_svm_pipeline(cfg: ClassifierConfig = ClassifierConfig()) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                max_features=cfg.max_features,
                ngram_range=cfg.ngram_range,
                min_df=cfg.min_df,
            )),
            ("model", LinearSVC()),
        ]
    )


def train_and_evaluate(
    df: pd.DataFrame,
    text_col: str = "text",
    target_col: str = "Therapeutic_Class",
    cfg: ClassifierConfig = ClassifierConfig(),
) -> Dict[str, object]:
    X = df[text_col]
    y = df[target_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    lr = make_lr_pipeline(cfg)
    svm = make_svm_pipeline(cfg)

    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    svm.fit(X_train, y_train)
    pred_svm = svm.predict(X_test)

    results = {
        "lr_report": classification_report(y_test, pred_lr, zero_division=0),
        "svm_report": classification_report(y_test, pred_svm, zero_division=0),
        "lr_model": lr,
        "svm_model": svm,
        "X_test": X_test,
        "y_test": y_test,
        "pred_lr": pred_lr,
        "pred_svm": pred_svm,
    }
    return results
