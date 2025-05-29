# Course-paper
Этот репозиторий был создан для работы над курсовой работой на тему "Бинарная классификация текстовых и спам-SMS-сообщений на основе готового набора данных".

# Инструкция по запуску проекта

Для корректного запуска проекта выполните следующие шаги:

1. **Установите зависимости**: pip install -r requirements.txt

2. **Запустите Jupyter**:  jupyter notebook
   

3. **Обучите модель**: python src/train.py
      

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


├── data
│   └── raw
│       └── Spam_SMS.csv
├── figures
│   ├── class_distribution.png
│   ├── confusion_matrix_Logistic Regression.png
│   ├── confusion_matrix_Naive Bayes.png
│   ├── confusion_matrix_Random Forest.png
│   ├── confusion_matrix_SVM.png
│   ├── confusion_matrix_XGBoost.png
│   └── message_length_distribution.png
├── models
│   ├── best_spam_model.pkl
│   └── spam_classifier.pkl
├── notebooks
│   └── 2_Model_Experiments.ipynb
├── README.md
├── reports
│   └── model_training.log
├── requirements.txt
├── results
│   └── metrics_comparison.md
└── src
    ├── __pycache__
    │   ├── config.cpython-313.pyc
    │   ├── config.cpython-39.pyc
    │   ├── data_preprocessing.cpython-313.pyc
    │   ├── evaluate.cpython-313.pyc
    │   ├── predict.cpython-313.pyc
    │   └── results_analysis.cpython-313.pyc
    ├── config.py
    ├── data_preprocessing.py
    ├── results_analysis.py
    └── train.py
