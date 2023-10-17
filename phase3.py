import pandas as pd

# Load the dataset into a pandas DataFrame
data = pd.read_csv('true.csv')
data = pd.read_csv('fake.csv')

# Check the structure of the dataset
print(data.head())
import re

def clean_text(text):
    # Remove non-alphanumeric characters and extra whitespaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

data['text'] = data['text'].apply(clean_text)
data['text'] = data['text'].str.lower()
from nltk.tokenize import word_tokenize

data['text'] = data['text'].apply(word_tokenize)
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

data['text'] = data['text'].apply(remove_stopwords)
data['label'] = data['label'].map({'fake': 0, 'real': 1})
from sklearn.model_selection import train_test_split

X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


