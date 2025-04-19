# Course-paper
This repository was created to work on a course paper on the topic "Binary classification of text and spam SMS messages on a ready-made dataset"


# Инструкция по запуску проекта

Для корректного запуска проекта выполните следующие шаги:

1. **Установите зависимости**: pip install -r requirements.txt

2. **Запустите EDA в Jupyter**:  jupyter notebook
   

3. **Обучите модель**: python src/train.py
   

4. **Протестируйте предсказания**: python src/predict.py
   

## Используемые модели

В рамках нашего проекта по классификации спама было применено несколько популярных алгоритмов машинного обучения. Ниже приведены описания каждой из моделей, использованных в эксперименте:

1. **Логистическая регрессия (Logistic Regression)**  
   
   LogisticRegression(maxiter=1000, randomstate=SEED)
   
  
2. **Случайный лес (Random Forest)**  
   
   RandomForestClassifier(nestimators=100, randomstate=SEED)
   
  
3. **XGBoost**  
   
   XGBClassifier(evalmetric='logloss', randomstate=SEED)
   
  
4. **Метод опорных векторов (SVM)**  
   
   SVC(kernel='linear', probability=True, random_state=SEED)
   
  
5. **Наивный байесовский классификатор (Naive Bayes)**  
   
   MultinomialNB()
   
  
Каждая из этих моделей была протестирована и оценена с использованием различных метрик, чтобы определить наилучший подход к решению задачи классификации спама.


# Структура проекта


spam_classification_project/<br>
│<br>
├── data/<br>
│   ├── raw/<br>
│   │   └── sms_spam.csv<br>
│   ├── processed/<br>
│   └── splits/<br>
│<br>
├── models/<br>
│   ├── baseline_model.pkl<br>
│   └── best_model.pkl<br>
│<br>
├── notebooks/<br>
│   ├── 1_EDA.ipynb<br>
│   └── 2_Model_Experiments.ipynb<br>
│<br>
├── src/<br>
│   ├── config.py<br>
│   ├── data_preprocessing.py<br>
│   ├── features.py<br>
│   ├── train.py<br>
│   ├── evaluate.py<br>
│   └── predict.py<br>
│<br>
├── reports/<br>
│   ├── figures/<br>
│   └── performance_metrics.txt<br>
│<br>
├── requirements.txt<br>
└── README.md<br>
