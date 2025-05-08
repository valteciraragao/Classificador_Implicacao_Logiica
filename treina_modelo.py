# treina_modelo.py
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Importe X, y do dataset_expressões.txt
X, y = [], []
with open("dataset_expressões.txt") as f:
    for line in f:
        expr, label = line.strip().split("\t")
        X.append(expr)
        y.append(label)

# Define tokenizer (mesma do app)
import re
def tokenizer(expr):
    return re.findall(r'->|\w+|[~&|()¬→]', expr)

# Cria e treina pipeline
vec = CountVectorizer(tokenizer=tokenizer, token_pattern=None)
clf = make_pipeline(vec, MultinomialNB())
clf.fit(X, y)

# Salva o novo modelo
joblib.dump(clf, "ml_inferencia_ampliado.joblib")
print("Modelo amplamente treinado e salvo em ml_inferencia_ampliado.joblib")
