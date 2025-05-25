from flask import Flask, render_template, request
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os
import re
# Setup nltk_data path
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Initialize Flask app
app = Flask(__name__)

# Load NLTK stopwords
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
try:
    model = joblib.load('sentiment_model.joblib')
    vectorizer = joblib.load('text_vectorizer.joblib')
except FileNotFoundError:
    model = None
    vectorizer = None
    print("Warning: Model files not found. Sentiment analysis will not work.")

# Text preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    result_class = ""

    if request.method == 'POST':
        text = request.form['Text'].strip()
        
        if not text:
            sentiment = "Please enter some text to analyze."
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

            if sentiment_label == "Positive":
                result_class = "positive-result"
            elif sentiment_label == "Negative":
                result_class = "negative-result"
            elif sentiment_label == "Neutral":
                result_class = "neutral-result"
            else:
                result_class = "info-message"
        else:
            sentiment = "Model files not loaded. Please contact the administrator."
            result_class = "info-message"

    return render_template('index.html', sentiment=sentiment, result_class=result_class)

# Run app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
