import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.phone_pattern = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')

    def extract_features(self, text):
        features = {
            'length': len(text),
            'num_digits': sum(c.isdigit() for c in text),
            'num_special': len(re.findall(r'[^a-zA-Z0-9\s]', text)),
            'has_url': 1 if self.url_pattern.search(text) else 0,
            'has_phone': 1 if self.phone_pattern.search(text) else 0
        }
        return features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        features = []
        for text in X:
            feat = self.extract_features(text)
            features.append(feat)
        
        return pd.DataFrame(features)

# Пример использования:
# feature_extractor = FeatureExtractor()
# additional_features = feature_extractor.transform(df['Message'])
# X = pd.concat([vectorized_text, additional_features], axis=1)