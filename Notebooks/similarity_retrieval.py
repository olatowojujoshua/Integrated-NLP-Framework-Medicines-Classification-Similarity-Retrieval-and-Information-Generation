from src.preprocessing import prepare_mid
from src.similarity import build_retrieval_index, query_by_name_contains, query_by_free_text

df = prepare_mid("data/MID.xlsx")
vectorizer, X_vec = build_retrieval_index(df)

chosen, candidates, similar = query_by_name_contains(df, vectorizer, X_vec, "Paracetamol", top_k=10)
chosen, candidates, similar

query_by_free_text(df, vectorizer, X_vec, "medicine used for hypertension and blood pressure", top_k=10)
