from src.preprocessing import prepare_mid
from src.summarization import MedicineSummarizer

df = prepare_mid("data/MID.xlsx")
summ = MedicineSummarizer()
summ.summarize_by_name(df, "Paracetamol O Tablet")
