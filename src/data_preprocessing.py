import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        # Удаляем всё, кроме букв и апострофов (для слов типа "don't")
        text = re.sub(r"[^a-zA-Z']", ' ', text)
        text = text.lower()
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) 
                for word in words if word not in self.stop_words]
        return ' '.join(words)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return [self.clean_text(text) for text in X]
    
    def clean_single_text(self, text):
        return self.clean_text(text)