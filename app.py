import os
import warnings

# Hide all TensorFlow and oneDNN logs completely
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings('ignore')

from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Initialize Flask application
app = Flask(__name__)

# Load the trained model and tokenizer from the EDA directory
model = load_model(r'EDA/spam_sms_model.h5', compile=False)

with open(r'EDA/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Setup NLTK tools for text preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()                                       # Convert text to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text)                    # Remove special characters and digits
    tokens = word_tokenize(text)                              # Tokenize the message into words

    cleaned_tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(cleaned_tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']

        # Clean and tokenize the input message
        cleaned_msg = clean_text(message)
        sequence = tokenizer.texts_to_sequences([cleaned_msg])
        padded = pad_sequences(
            sequence,
            maxlen=50,
            padding='post',
            truncating='post'
        )

        # Predict probability using the model
        prediction = model.predict(padded, verbose=0)
        score = prediction[0][0]

        # Format the result based on classification threshold and print in terminal
        if score > 0.5:
            prediction_text = f"🚨 SPAM Message Detected! (Confidence: {score * 100:.2f}%)"
            result_class = "spam"
            print(f"\n[BROWSER CHECK] Message: '{message}' ==> Result: SPAM (Score: {score:.4f})")
        else:
            prediction_text = f"✅ HAM / Normal Message (Confidence: {(1 - score) * 100:.2f}%)"
            result_class = "ham"
            print(f"\n[BROWSER CHECK] Message: '{message}' ==> Result: HAM (Score: {score:.4f})")

        return render_template(
            'index.html',
            prediction_text=prediction_text,
            result_class=result_class,
            message=message
        )

if __name__ == '__main__':
    # Run Flask application cleanly
    app.run(
        debug=False,
        use_reloader=False
    )