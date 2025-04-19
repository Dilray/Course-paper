# Course-paper
This repository was created to work on a course paper on the topic "Binary classification of text and spam SMS messages on a ready-made dataset"


# Инструкция по запуску проекта

Для корректного запуска проекта выполните следующие шаги:

1. **Установите зависимости**: pip install -r requirements.txt

2. **Запустите EDA в Jupyter**:  jupyter notebook
   

3. **Обучите модель**: python src/train.py
   

4. **Протестируйте предсказания**: python src/predict.py
   



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