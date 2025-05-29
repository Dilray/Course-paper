import joblib
import pandas as pd

class SpamClassifier:
    def __init__(self, model_path='../models/best_model.pkl'):
        self.model = joblib.load(model_path)
    
    def predict(self, messages):
        if isinstance(messages, str):
            messages = [messages]
        
        # Преобразование в DataFrame для удобства
        df = pd.DataFrame({'Message': messages})
        predictions = self.model.predict(df['Message'])
        probabilities = self.model.predict_proba(df['Message'])
        
        results = []
        for msg, pred, prob in zip(messages, predictions, probabilities):
            results.append({
                'message': msg,
                'prediction': 'spam' if pred == 1 else 'ham',
                'spam_probability': prob[1],
                'ham_probability': prob[0]
            })
        
        return results

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='../reports/spam_classifier.log'
)
logger = logging.getLogger(__name__)