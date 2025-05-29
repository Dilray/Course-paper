import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import logging
import numpy as np
from config import SEED, TEST_SIZE, MODELS_DIR, REPORTS_DIR, FIGURES_DIR
from data_preprocessing import TextPreprocessor

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=REPORTS_DIR / 'model_training.log'
)
logger = logging.getLogger(__name__)

def load_data():
    """Загрузка данных"""
    df = pd.read_csv("/Users/egorkulishov/Course-paper/data/raw/Spam_SMS.csv")
    df['label'] = df['Class'].map({'spam': 1, 'ham': 0})
    return train_test_split(
        df['Message'], 
        df['label'], 
        test_size=TEST_SIZE, 
        random_state=SEED
    )

def evaluate_models(X_train, y_train):
    """Тестирование нескольких моделей с кросс-валидацией"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=SEED, max_iter=1000),
        'SVM': SVC(kernel='linear', probability=True, random_state=SEED),
        'Random Forest': RandomForestClassifier(random_state=SEED),
        'XGBoost': XGBClassifier(random_state=SEED, eval_metric='logloss'),
        'Naive Bayes': MultinomialNB()
    }

    results = {}
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', TextPreprocessor()),
            ('vectorizer', CountVectorizer(max_features=5000)),
            ('classifier', model)
        ])
        
        # Кросс-валидация (5-fold)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
        results[name] = {
            'model': pipeline,
            'cv_mean_f1': np.mean(cv_scores),
            'cv_std': np.std(cv_scores)
        }
        logger.info(f"{name}: F1 = {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    
    return results

def train_best_model(results, X_train, y_train):
    """Обучение лучшей модели на всех данных"""
    best_model_name = max(results, key=lambda x: results[x]['cv_mean_f1'])
    best_pipeline = results[best_model_name]['model']
    
    # Фитируем на всех тренировочных данных
    best_pipeline.fit(X_train, y_train)
    
    # Сохраняем модель
    model_path = MODELS_DIR / 'best_spam_model.pkl'
    joblib.dump(best_pipeline, model_path)
    logger.info(f"Best model ({best_model_name}) saved to {model_path}")
    
    return best_pipeline


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    
    # Тестируем модели
    model_results = evaluate_models(X_train, y_train)
    
    # Обучаем лучшую модель
    best_model = train_best_model(model_results, X_train, y_train)
    
    # Финальная оценка
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])
    logger.info("\nFinal Classification Report:\n" + report)