import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
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
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=SEED),
        'SVM': SVC(kernel='linear', probability=True, random_state=SEED),  # Новая модель
        'Naive Bayes': MultinomialNB()  # Новая модель
    }
    
    best_model = None
    best_f1 = 0
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        report = classification_report(y_test, y_pred)
        print(f"{name} results:\n{report}")
        
        f1 = float(report.split()[-4])  # Извлекаем F1-score
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            print(f"New best model: {name} with F1={f1:.4f}")
    
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    print(f"\nBest model ({type(best_model).__name__}) saved to 'models/best_model.pkl'")

if __name__ == '__main__':
    train()