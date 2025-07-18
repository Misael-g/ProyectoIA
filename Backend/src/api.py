# src/api.py
from flask import Flask, request, jsonify
import pickle
from utils import limpiar_texto
import os

app = Flask(__name__)

# Cargar modelo
model_path = '../models/sentiment_model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No se encontró el modelo en {model_path}. Primero ejecuta train.py")

vectorizer, model = pickle.load(open(model_path, 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    X = vectorizer.transform([limpiar_texto(text)])
    pred = int(model.predict(X)[0])
    return jsonify({'prediction': pred})

if __name__ == '__main__':
    app.run(port=5000)
