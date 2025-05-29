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
import os
from config import SEED, TEST_SIZE
from results_analysis import (plot_class_distribution, 
                             plot_message_length_distribution,
                             plot_confusion_matrix,
                             generate_metrics_table)

def train():
    df = pd.read_csv('data/raw/Spam_SMS.csv')
    df['label'] = df['Class'].map({'ham': 0, 'spam': 1})

    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Визуализация распределения классов и длины сообщений
    plot_class_distribution(df['label'])
    plot_message_length_distribution(df, text_column='Message', target_column='label')

    X_train, X_test, y_train, y_test = train_test_split(
        df['Message'], df['label'], test_size=TEST_SIZE, random_state=SEED)
    
    # Векторизация текста
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Инициализация моделей
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=SEED),
        'SVM': SVC(kernel='linear', probability=True, random_state=SEED),
        'Naive Bayes': MultinomialNB()
    }
    
    # Обучение и оценка моделей
    best_model = None
    best_f1 = 0
    trained_models = {}  # Для хранения обученных моделей
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        # Сохранение обученной модели
        trained_models[name] = model
        
        # Матрица ошибок для каждой модели
        plot_confusion_matrix(y_test, y_pred, name)
        
        report = classification_report(y_test, y_pred)
        print(f"{name} results:\n{report}")
        
        f1 = float(report.split()[-4])  # Извлекаем F1-score
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            print(f"New best model: {name} with F1={f1:.4f}")
    
    # Генерация сводной таблицы метрик
    generate_metrics_table(trained_models, X_test_vec, y_test)
    
    # Сохранение лучшей модели и векторизатора
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    print(f"\nBest model ({type(best_model).__name__}) saved to 'models/best_model.pkl'")

if __name__ == '__main__':
    train()