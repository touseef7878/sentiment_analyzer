from flask import Flask, render_template, request
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load the trained model and vectorizer
try:
    model = joblib.load('sentiment_model.joblib')
    vectorizer = joblib.load('text_vectorizer.joblib')
except FileNotFoundError:
    model = None
    vectorizer = None
    print("Warning: Model files not found. Sentiment analysis will not work.")

# Download NLTK resources if not already present
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    if request.method == 'POST':
        text = request.form['Text']
        if model and vectorizer:
            processed_text = preprocess_text(text)
            vectorized_text = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized_text)[0]
            sentiment_map = {1: "Positive", 0: "Negative", 2: "Neutral"} # Adjust if your mapping is different
            sentiment = sentiment_map.get(prediction, "Unknown")
        else:
            sentiment = "Model not loaded."
    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)