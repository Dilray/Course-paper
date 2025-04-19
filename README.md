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
   
python
   LogisticRegression(maxiter=1000, randomstate=SEED)
   
  
   Логистическая регрессия — это статистический метод, используемый для бинарной классификации. Было настроено максимальное количество итераций на 1000 для достижения сходимости и установили фиксированное состояние генератора случайных чисел для воспроизводимости результатов.

2. **Случайный лес (Random Forest)**  
   
python
   RandomForestClassifier(nestimators=100, randomstate=SEED)
   
  
   Случайный лес представляет собой ансамблевый метод, который использует множество деревьев решений для улучшения точности и контроля за переобучением. В нашем случае я использовал 100 деревьев, что позволяет достичь хорошего баланса между производительностью и вычислительной сложностью.

3. **XGBoost**  
   
python
   XGBClassifier(evalmetric='logloss', randomstate=SEED)
   
  
   XGBoost (Extreme Gradient Boosting) — это мощный алгоритм градиентного бустинга, который часто демонстрирует отличные результаты на задачах классификации. Я использовал метрику логарифмической потери (logloss) для оценки качества модели во время обучения.

4. **Метод опорных векторов (SVM)**  
   
python
   SVC(kernel='linear', probability=True, random_state=SEED)
   
  
   Метод опорных векторов — это мощный инструмент для классификации, который ищет оптимальную гиперплоскость для разделения классов. Я выбрал линейное ядро и включил возможность предсказания вероятностей.

5. **Наивный байесовский классификатор (Naive Bayes)**  
   
python
   MultinomialNB()
   
  
   Наивный байесовский классификатор — это простой и эффективный алгоритм, основанный на применении теоремы Байеса с предположением о независимости признаков. Он особенно хорошо работает с текстовыми данными, такими как сообщения электронной почты.

Каждая из этих моделей была протестирована и оценена с использованием различных метрик, чтобы определить наилучший подход к решению задачи классификации спама.


# Структура проекта


spam_classification_project/
│
├── data/
│   ├── raw/
│   │   └── sms_spam.csv
│   ├── processed/
│   └── splits/
│
├── models/
│   ├── baseline_model.pkl
│   └── best_model.pkl
│
├── notebooks/
│   ├── 1_EDA.ipynb
│   └── 2_Model_Experiments.ipynb
│
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── reports/
│   ├── figures/
│   └── performance_metrics.txt
│
├── requirements.txt
└── README.md