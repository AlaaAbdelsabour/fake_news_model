from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

app = Flask(_name_)

# تحميل الموديل والتوكنيزر
model = tf.keras.models.load_model('arabic_fake_news_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

MAX_LEN = 200
label_map = {0: 'not credible', 1: 'undecided', 2: 'credible'}

def clean_arabic(text):
    text = str(text)
    text = re.sub(r'http\S+|www.\S+|@\w+|#\w+', '', text)
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\s،؛؟.!]', '', text)
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'[ة]', 'ه', text)
    return re.sub(r'\s+', ' ', text).strip()

@app.route('/')
def index():
    return "Arabic Fake News Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    input_text = clean_arabic(data['text'])
    seq = tokenizer.texts_to_sequences([input_text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

    prediction = model.predict(padded)
    predicted_class = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_class])
    label = label_map[predicted_class]

    return jsonify({'prediction': label, 'confidence': confidence})

if _name_ == '_main_':
    app.run(debug=True)