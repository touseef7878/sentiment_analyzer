import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data if not already present
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

# Load the dataset
try:
    data = pd.read_csv('sentiment_data.csv', encoding='latin-1')  # use utf-8 if needed
    data = data[['sentiment', 'text']]  # keep only the relevant columns
    data = data.dropna()
except FileNotFoundError:
    print("Error: sentiment_data.csv not found. Please make sure the file exists.")
    exit()

# Convert sentiment labels to numbers
sentiment_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}
data['sentiment'] = data['sentiment'].map(sentiment_mapping)

# Preprocess text column
data['processed_text'] = data['text'].apply(preprocess_text)

# Split data
X = data['processed_text']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Evaluate model
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(vectorizer, 'text_vectorizer.joblib')

print("✅ Trained sentiment model and vectorizer saved.")