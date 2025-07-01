import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import joblib
import nltk
import re

# Download NLTK resources
nltk.download('punkt')

# Text preprocessing function (keep stopwords for context)
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load dataset
try:
    data = pd.read_csv('sentiment_data.csv', encoding='latin-1')
    data = data[['sentiment', 'text']].dropna()
except FileNotFoundError:
    print("Error: sentiment_data.csv not found.")
    exit()

# Map labels
sentiment_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}
data['sentiment'] = data['sentiment'].map(sentiment_mapping)

# Preprocess
data['processed_text'] = data['text'].apply(preprocess_text)

# Handle class imbalance (downsample neutral)
positive = data[data.sentiment == 1]
negative = data[data.sentiment == 0]
neutral = data[data.sentiment == 2]
neutral_downsampled = resample(neutral, replace=False, n_samples=min(len(positive), len(negative)), random_state=42)

# Combine
balanced_data = pd.concat([positive, negative, neutral_downsampled])

# Split
X = balanced_data['processed_text']
y = balanced_data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize with tuned parameters
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=5)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectorized, y_train)

# Evaluate
y_pred = model.predict(X_test_vectorized)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive', 'Neutral']))

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(vectorizer, 'text_vectorizer.joblib')
print("✅ Trained model and vectorizer saved.")
