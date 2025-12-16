from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass(frozen=True)
class MIDColumns:
    name: str = "Name"
    link: str = "Link"
    contains: str = "Contains"
    target: str = "Therapeutic_Class"

    text_cols: tuple[str, ...] = (
        "ProductIntroduction",
        "ProductUses",
        "ProductBenefits",
        "SideEffect",
        "HowToUse",
        "HowWorks",
        "QuickTips",
        "SafetyAdvice",
        # optional boosters:
        # "Chemical_Class",
        # "Action_Class",
        # "Habit_Forming",
        # "Contains",
    )


def clean_text(s: object) -> str:
    """Lightweight, fast cleaner suitable for TF-IDF and retrieval."""
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"[^a-z0-9\s\-\+\/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_mid_excel(path: str) -> pd.DataFrame:
    """Loads MID.xlsx into a DataFrame."""
    df = pd.read_excel(path)
    return df


def dedupe_mid(df: pd.DataFrame, cols: MIDColumns = MIDColumns()) -> pd.DataFrame:
    """Remove exact duplicates (best practice for retrieval)."""
    keep_cols = [c for c in [cols.name, cols.link] if c in df.columns]
    if keep_cols:
        df = df.drop_duplicates(subset=keep_cols)
    return df.reset_index(drop=True)


def build_text_fields(
    df: pd.DataFrame,
    cols: MIDColumns = MIDColumns(),
    text_col_name: str = "text",
    text_raw_col_name: str = "text_raw",
) -> pd.DataFrame:
    """Create text_raw (joined) and text (cleaned) columns."""
    missing = [c for c in cols.text_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected text columns: {missing}")

    df = df.copy()
    df[text_raw_col_name] = df[list(cols.text_cols)].fillna("").astype(str).agg(" ".join, axis=1)
    df[text_col_name] = df[text_raw_col_name].apply(clean_text)
    return df


def filter_valid_targets(
    df: pd.DataFrame, cols: MIDColumns = MIDColumns()
) -> pd.DataFrame:
    """Remove rows with empty/NaN targets."""
    if cols.target not in df.columns:
        raise ValueError(f"Target column '{cols.target}' not found in dataframe.")

    out = df.copy()
    out[cols.target] = out[cols.target].astype(str)
    mask = out[cols.target].notna() & (out[cols.target].str.strip() != "")
    return out.loc[mask].reset_index(drop=True)


def prepare_mid(
    mid_xlsx_path: str,
    cols: MIDColumns = MIDColumns(),
) -> pd.DataFrame:
    """One-call full prep: load -> dedupe -> build text -> filter targets."""
    df = load_mid_excel(mid_xlsx_path)
    df = dedupe_mid(df, cols=cols)
    df = build_text_fields(df, cols=cols)
    df = filter_valid_targets(df, cols=cols)
    return df
