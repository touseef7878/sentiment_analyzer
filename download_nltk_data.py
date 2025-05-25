import nltk
import os

# Local nltk_data path
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)

# Set NLTK path
nltk.data.path.append(nltk_data_path)

# Download resources locally
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

print("✅ NLTK data downloaded locally to:", nltk_data_path)
