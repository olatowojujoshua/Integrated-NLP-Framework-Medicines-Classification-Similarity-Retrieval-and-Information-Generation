from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RetrievalConfig:
    max_features: int = 90000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    sublinear_tf: bool = True


def build_retrieval_index(
    df: pd.DataFrame,
    text_col: str = "text",
    cfg: RetrievalConfig = RetrievalConfig(),
):
    vectorizer = TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        sublinear_tf=cfg.sublinear_tf,
        norm="l2",
    )
    X_vec = vectorizer.fit_transform(df[text_col])
    return vectorizer, X_vec


def get_candidates(
    df: pd.DataFrame,
    query: str,
    name_col: str = "Name",
    limit: int = 20,
) -> pd.DataFrame:
    return df[df[name_col].astype(str).str.contains(query, case=False, na=False)][[name_col, "Therapeutic_Class"]].head(limit)


def smart_anchor_index(
    df: pd.DataFrame,
    query: str,
    name_col: str = "Name",
) -> int:
    cand = df[df[name_col].astype(str).str.contains(query, case=False, na=False)].copy()
    if cand.empty:
        raise ValueError(f"No drug containing '{query}' found")

    q = query.strip().lower()
    # rank: whole-word contains first, then shorter names
    cand["_rank"] = cand[name_col].astype(str).str.lower().apply(
        lambda n: (0 if f" {q} " in f" {n} " else 1, len(n))
    )
    cand = cand.sort_values("_rank").drop(columns="_rank")
    return int(cand.index[0])


def top_k_similar(
    df: pd.DataFrame,
    X_vec,
    anchor_idx: int,
    top_k: int = 10,
    unique_by_link: bool = True,
    cols_to_show: tuple[str, ...] = ("Name", "Therapeutic_Class", "Link", "Contains"),
) -> pd.DataFrame:
    sims = cosine_similarity(X_vec[anchor_idx], X_vec).flatten()
    ranked = np.argsort(sims)[::-1]

    picked = []
    seen_links = set()

    for j in ranked:
        if j == anchor_idx:
            continue
        if unique_by_link and "Link" in df.columns:
            link = df.loc[j, "Link"]
            if link in seen_links:
                continue
            seen_links.add(link)
        picked.append(j)
        if len(picked) >= top_k:
            break

    out = df.loc[picked, list(cols_to_show)].copy()
    out["similarity"] = sims[picked]
    return out.reset_index(drop=True)


def query_by_name_contains(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    X_vec,
    query: str,
    top_k: int = 10,
    name_col: str = "Name",
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    anchor_idx = smart_anchor_index(df, query, name_col=name_col)
    chosen = df.loc[anchor_idx, [name_col, "Therapeutic_Class", "Contains", "Link"]].to_dict()
    candidates = get_candidates(df, query, name_col=name_col, limit=10)
    similar = top_k_similar(df, X_vec, anchor_idx, top_k=top_k)
    return chosen, candidates, similar


def query_by_free_text(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    X_vec,
    query: str,
    top_k: int = 10,
    cols_to_show: tuple[str, ...] = ("Name", "Therapeutic_Class", "Link", "Contains"),
) -> pd.DataFrame:
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X_vec).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    out = df.loc[top_idx, list(cols_to_show)].copy()
    out["similarity"] = sims[top_idx]
    return out.reset_index(drop=True)
