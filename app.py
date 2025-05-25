from flask import Flask, render_template, request
import joblib
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set up local nltk_data path
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Initialize Flask app
app = Flask(__name__)

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
try:
    model = joblib.load('sentiment_model.joblib')
    vectorizer = joblib.load('text_vectorizer.joblib')
except FileNotFoundError:
    model = None
    vectorizer = None
    print("Model files not found.")

# Text preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    result_class = ""

    if request.method == 'POST':
        text = request.form['Text'].strip()

        if not text:
            sentiment = "Please enter some text."
            result_class = "info-message"
        elif model and vectorizer:
            processed_text = preprocess_text(text)
            vectorized_text = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized_text)[0]

            sentiment_map = {1: "Positive", 0: "Negative", 2: "Neutral"}
            sentiment_label = sentiment_map.get(prediction, "Unknown")

            probabilities = model.predict_proba(vectorized_text)[0]
            confidence = probabilities[list(model.classes_).index(prediction)] * 100
            sentiment = f"{sentiment_label}: {confidence:.2f}%"

            result_class = f"{sentiment_label.lower()}-result"
        else:
            sentiment = "Model files not loaded."
            result_class = "info-message"

    return render_template('index.html', sentiment=sentiment, result_class=result_class)

if __name__ == '__main__':
    app.run(debug=True)
# Ensure nltk resources are downloaded