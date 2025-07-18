import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import limpiar_texto
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# 1. Cargar dataset limpio
df = pd.read_csv('../data/reviews_limpio.csv', encoding='utf-8')

# 2. Variables independientes y dependientes
X = df['clean_text'].astype(str)
y = df['label'].values

# 3. Verificar que no haya nulos
assert not df.isnull().values.any(), "El dataset contiene valores nulos"

# 4. Dividir el conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Tokenización y padding
vocab_size = 10000
max_len = 100

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# 6. Crear modelo con Keras
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Entrenar modelo
# Definir early stopping para monitorizar la pérdida de validación (val_loss)
early_stop = EarlyStopping(
    monitor='val_loss',  
    patience=2,         
    verbose=1,             
    restore_best_weights=True 
)

# Entrenamiento con early stopping
model.fit(
    X_train_pad, y_train,
    epochs=20,             
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# 8. Evaluar modelo
loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)
y_pred = (model.predict(X_test_pad) > 0.5).astype("int32")

print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("📈 Classification Report:\n", classification_report(y_test, y_pred))

# 9. Guardar modelo y tokenizer
model.save('../models/sentiment_nn_model.h5')
with open('../models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("\n✅ Modelo guardado como modelo h5 y tokenizer en pkl")

# 10. Probar con frases reales
test_phrases = [
    "El producto es excelente y me encantó",
    "Muy decepcionado, no funciona como esperaba",
    "La calidad es pésima, no lo recomiendo",
    "Servicio al cliente fantástico y rápido",
    "No me gustó, llegó dañado y mal embalado",
    "Estoy muy satisfecho con la compra, la recomiendo",
    "Terrible experiencia, no volveré a comprar",
    "Muy buena calidad, vale la pena",
    "El peor producto que he comprado",
    "Cumple con lo que promete, muy bien"
]

test_phrases_clean = [limpiar_texto(p) for p in test_phrases]
test_seq = tokenizer.texts_to_sequences(test_phrases_clean)
test_pad = pad_sequences(test_seq, maxlen=max_len)
preds = model.predict(test_pad)

print("\n📝 Clasificación de ejemplos realistas:")
for phrase, pred in zip(test_phrases, preds):
    print(f"  '{phrase}': {'Positivo 😊' if pred >= 0.5 else 'Negativo 😞'}")
