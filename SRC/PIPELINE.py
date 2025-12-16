from __future__ import annotations

import os
import json
from typing import Dict, Any

import pandas as pd

from src.preprocessing import prepare_mid, MIDColumns, clean_text
from src.classification import make_lr_pipeline, make_svm_pipeline, ClassifierConfig
from src.similarity import (
    build_retrieval_index,
    query_by_name_contains,
    query_by_free_text,
)
from src.summarization import MedicineSummarizer


class MedicinesPipeline:
    """
    Unified pipeline for:
    - Therapeutic class classification
    - Similarity retrieval
    - Medicine summarization
    """

    def __init__(
        self,
        mid_path: str = "data/MID.xlsx",
        use_sample: bool = False,
        sample_size: int = 60000,
        enable_summarization: bool = True,
    ):
        self.cols = MIDColumns()
        self.mid_path = mid_path
        self.use_sample = use_sample
        self.sample_size = sample_size
        self.enable_summarization = enable_summarization

        self.df: pd.DataFrame | None = None
        self.vectorizer = None
        self.X_vec = None
        self.lr_model = None
        self.svm_model = None
        self.summarizer = None

        self._load_and_prepare()
        self._build_retrieval()
        self._train_classifiers()

        if self.enable_summarization:
            self.summarizer = MedicineSummarizer()

   
    # Internal setup methods
 
    def _load_and_prepare(self):
        self.df = prepare_mid(self.mid_path, cols=self.cols)

        if self.use_sample:
            self.df = self.df.sample(
                n=min(self.sample_size, len(self.df)),
                random_state=42,
            ).reset_index(drop=True)

    def _build_retrieval(self):
        self.vectorizer, self.X_vec = build_retrieval_index(self.df)

    def _train_classifiers(self):
        cfg = ClassifierConfig()
        self.lr_model = make_lr_pipeline(cfg)
        self.svm_model = make_svm_pipeline(cfg)

        X = self.df["text"]
        y = self.df[self.cols.target].astype(str)

        self.lr_model.fit(X, y)
        self.svm_model.fit(X, y)


    # Public pipeline methods
   
    def classify_by_name(self, drug_name: str) -> Dict[str, Any]:
        row = self.df[self.df[self.cols.name].astype(str).str.lower() == drug_name.lower()]
        if row.empty:
            return {"error": f"Drug not found: {drug_name}"}

        row = row.iloc[0]
        text = row["text"]

        return {
            "Name": row[self.cols.name],
            "Actual_Therapeutic_Class": row[self.cols.target],
            "Predicted_LinearSVC": self.svm_model.predict([text])[0],
            "Predicted_LogisticRegression": self.lr_model.predict([text])[0],
        }

    def classify_by_text(self, free_text: str) -> Dict[str, Any]:
        clean = clean_text(free_text)
        return {
            "Predicted_LinearSVC": self.svm_model.predict([clean])[0],
            "Predicted_LogisticRegression": self.lr_model.predict([clean])[0],
        }

    def retrieve_similar_by_name(self, query: str, top_k: int = 10):
        chosen, candidates, similar = query_by_name_contains(
            df=self.df,
            vectorizer=self.vectorizer,
            X_vec=self.X_vec,
            query=query,
            top_k=top_k,
        )
        return {
            "anchor": chosen,
            "candidates": candidates,
            "similar": similar,
        }

    def retrieve_similar_by_text(self, query: str, top_k: int = 10):
        return query_by_free_text(
            df=self.df,
            vectorizer=self.vectorizer,
            X_vec=self.X_vec,
            query=clean_text(query),
            top_k=top_k,
        )

    def summarize_medicine(self, drug_name: str) -> Dict[str, Any]:
        if not self.enable_summarization:
            return {"error": "Summarization disabled"}

        return self.summarizer.summarize_by_name(self.df, drug_name)
