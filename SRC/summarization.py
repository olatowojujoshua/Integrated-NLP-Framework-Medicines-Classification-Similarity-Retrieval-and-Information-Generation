from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import pandas as pd
from transformers import pipeline


@dataclass(frozen=True)
class SummarizerConfig:
    model_name: str = "facebook/bart-large-cnn"
    max_chars: int = 3500
    max_length: int = 130
    min_length: int = 45
    do_sample: bool = False


class MedicineSummarizer:
    def __init__(self, cfg: SummarizerConfig = SummarizerConfig()):
        self.cfg = cfg
        self._summarizer = pipeline("summarization", model=cfg.model_name)

    def summarize_text(self, text: str) -> str:
        text = (text or "")[: self.cfg.max_chars]
        out = self._summarizer(
            text,
            max_length=self.cfg.max_length,
            min_length=self.cfg.min_length,
            do_sample=self.cfg.do_sample,
        )
        return out[0]["summary_text"]

    def summarize_by_name(
        self,
        df: pd.DataFrame,
        drug_name: str,
        name_col: str = "Name",
        therapeutic_col: str = "Therapeutic_Class",
        text_raw_col: str = "text_raw",
    ) -> Dict[str, str]:
        matches = df[df[name_col].astype(str).str.lower() == drug_name.lower()]
        if matches.empty:
            return {"error": f"Drug not found: {drug_name}"}

        idx = matches.index[0]
        summary = self.summarize_text(str(df.loc[idx, text_raw_col]))
        return {
            "Name": str(df.loc[idx, name_col]),
            "Therapeutic_Class": str(df.loc[idx, therapeutic_col]),
            "Summary": summary,
        }
