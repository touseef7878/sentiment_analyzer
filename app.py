from flask import Flask, render_template, request
import joblib
import nltk
import re
import os # Import os for deployment configuration
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

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
    result_class = ""
    if request.method == 'POST':
        text = request.form['Text'].strip() # .strip() removes leading/trailing whitespace

        if not text: 
            sentiment = "Please enter some text to analyze."
            result_class = "info-message" 
        elif model and vectorizer:
            processed_text = preprocess_text(text)
            vectorized_text = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized_text)[0]

            #  model predicts 0 for Negative, 1 for Positive, 2 for Neutral:
            sentiment_map = {1: "Positive", 0: "Negative", 2: "Neutral"}
            sentiment_label = sentiment_map.get(prediction, "Unknown")
            
            # Get probabilities for each class
            probabilities = model.predict_proba(vectorized_text)[0]
            
            # Find the index of the predicted class in model.classes_
            predicted_class_prob_index = list(model.classes_).index(prediction)
            confidence = probabilities[predicted_class_prob_index] * 100 
            sentiment = f"{sentiment_label}: {confidence:.2f}%"

            if sentiment_label == "Positive":
                result_class = "positive-result" # Default success style 
            elif sentiment_label == "Negative":
                result_class = "negative-result"
            elif sentiment_label == "Neutral":
                result_class = "neutral-result"
            else:
                result_class = "info-message" # For "Unknown" or other unexpected labels
        else:
            sentiment = "Model files not loaded. Please contact administrator."
            result_class = "info-message" # Assign class for info messages
    return render_template('index.html', sentiment=sentiment, result_class=result_class)

if __name__ == '__main__':
    # Use 0.0.0.0 and PORT from environment for deployment (like Render)
    # Use debug=True for local development
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)