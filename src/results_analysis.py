import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, recall_score, precision_score

def plot_class_distribution(y_true):
    """Гистограмма распределения классов (спам vs. не-спам)"""
    plt.figure(figsize=(8, 5))
    sns.countplot(x=y_true)
    plt.title("Распределение классов в датасете")
    plt.xlabel("Класс")
    plt.ylabel("Количество сообщений")
    plt.savefig("figures/class_distribution.png")  # Сохраняем график
    plt.close()

def plot_message_length_distribution(df, text_column='text', target_column='label'):
    """Box-plot распределения длины сообщений по классам"""
    df['length'] = df[text_column].apply(len)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=target_column, y='length', data=df)
    plt.title("Распределение длины сообщений по классам")
    plt.xlabel("Класс")
    plt.ylabel("Длина сообщения (символы)")
    plt.savefig("figures/message_length_distribution.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Матрица ошибок для одной модели"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Матрица ошибок: {model_name}")
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.savefig(f"figures/confusion_matrix_{model_name}.png")
    plt.close()

def generate_metrics_table(models, X_test, y_test):
    """Сводная таблица метрик для всех моделей"""
    metrics = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics.append({
            'Модель': name,
            'Точность (Precision)': precision_score(y_test, y_pred),
            'Полнота (Recall)': recall_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred),
            'Accuracy': accuracy_score(y_test, y_pred)
        })
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_markdown("results/metrics_comparison.md", index=False)  # Сохраняем в Markdown
    return df_metrics
