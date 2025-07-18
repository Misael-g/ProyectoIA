from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
from utils import limpiar_texto
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)

max_len = 100 

model_path = '../models/sentiment_nn_model.h5'
tokenizer_path = '../models/tokenizer.pkl'

if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
    raise FileNotFoundError("Modelo o tokenizer no encontrados. Ejecuta train.py primero.")

model = tf.keras.models.load_model(model_path)

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
  
    cleaned_text = limpiar_texto(text)
    
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=max_len)
    
    pred_prob = model.predict(padded)[0][0]
    
    pred_class = int(pred_prob >= 0.5)
    
    return jsonify({'prediction': pred_class, 'probability': float(pred_prob)})

if __name__ == '__main__':
    app.run(port=5000)
