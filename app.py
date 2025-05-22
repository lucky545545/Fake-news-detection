from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
import pickle

app = Flask(__name__)

try:
    # Download required NLTK data only if not already present
    nltk.download('stopwords')

    # Load the pre-trained model
    model = tf.keras.models.load_model('fake_new_det.keras')

    # Load the tokenizer used during training
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    max_features = 10000  # Used during training

except Exception as e:
    print(f"Error during initialization: {str(e)}")
    raise


# Text preprocessing functions
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    return text.lower()

def remove_between_square_brackets(text):
    return re.sub(r'http\S+', '', text)

def remove_stopwords(text):
    stop = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word.lower() not in stop])

def denoise_text(text):
    text = remove_between_square_brackets(text)
    text = clean_text(text)
    return remove_stopwords(text)

def preprocess(text):
    try:
        cleaned = denoise_text(text)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=300, padding='post', truncating='post')
        return padded
    except Exception as e:
        raise ValueError(f"Error in preprocessing: {str(e)}")


# Root route for basic browser access
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        if not data or 'news' not in data:
            return jsonify({
                'error': 'Missing news text in request body',
                'status': 400
            }), 400

        news_text = data['news']
        if not isinstance(news_text, str) or not news_text.strip():
            return jsonify({
                'error': 'Invalid news text provided',
                'status': 400
            }), 400

        processed = preprocess(news_text)
        prediction = model.predict(processed)

        return jsonify({
            'result': 'Fake' if prediction[0][0] < 0.5 else 'True',
            'confidence': float(prediction[0][0]),
            'status': 200
        })

    except ValueError as ve:
        return jsonify({
            'error': str(ve),
            'status': 400
        }), 400
    except Exception as e:
        return jsonify({
            'error': f"Internal server error: {str(e)}",
            'status': 500
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)