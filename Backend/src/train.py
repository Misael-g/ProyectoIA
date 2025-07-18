import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import limpiar_texto

# Leer dataset
df = pd.read_csv('../data/reviews.csv')
df['text'] = df['text'].apply(limpiar_texto)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Vectorizador PRO
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Entrenar modelo
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluación
preds = model.predict(X_test_vec)
print("📊 Accuracy:", accuracy_score(y_test, preds))
print("🧾 Confusion Matrix:\n", confusion_matrix(y_test, preds))
print("📈 Classification Report:\n", classification_report(y_test, preds))

# Guardar modelo
with open('../models/sentiment_model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)

print("✅ Modelo guardado en ../models/sentiment_model.pkl")

# 🚀 Test manual con frases comunes
test_phrases = [
    "excelente", "muy bueno", "horrible", "fantástico",
    "pésimo", "no me gustó", "exelente", "bno", "orrible"
]
X_test_manual = vectorizer.transform([limpiar_texto(p) for p in test_phrases])
preds_manual = model.predict(X_test_manual)

print("\n📝 Clasificación de ejemplos manuales:")
for phrase, pred in zip(test_phrases, preds_manual):
    print(f"  '{phrase}': {'Positivo 😊' if pred==1 else 'Negativo 😞'}")
