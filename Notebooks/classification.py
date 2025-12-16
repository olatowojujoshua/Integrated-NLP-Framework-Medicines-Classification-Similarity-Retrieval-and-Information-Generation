from src.preprocessing import prepare_mid
from src.classification import train_and_evaluate

df = prepare_mid("data/MID.xlsx")
results = train_and_evaluate(df)

print("TF-IDF + Logistic Regression\n", results["lr_report"])
print("TF-IDF + LinearSVC\n", results["svm_report"])


