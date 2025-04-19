import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from config import SEED, TEST_SIZE

def train():
    df = pd.read_csv('data/raw/Spam_SMS.csv')
    df['label'] = df['Class'].map({'ham': 0, 'spam': 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df['Message'], df['label'], test_size=TEST_SIZE, random_state=SEED)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred))
    
    joblib.dump(model, 'models/baseline_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')

if __name__ == '__main__':
    train()