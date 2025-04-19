import joblib
import pandas as pd
from sklearn.metrics import (classification_report, 
                            confusion_matrix,
                            roc_auc_score,
                            accuracy_score,
                            f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
from config import REPORTS_DIR, FIGURES_DIR

def evaluate_model(model_path, X_test, y_test):
    model = joblib.load(model_path)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Вероятности для класса spam
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics_file = REPORTS_DIR / 'performance_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['ham', 'spam'],
                yticklabels=['ham', 'spam'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(FIGURES_DIR / 'confusion_matrix.png')
    plt.close()
    
    from sklearn.metrics import RocCurveDisplay
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title('ROC Curve')
    plt.savefig(FIGURES_DIR / 'roc_curve.png')
    plt.close()
    
    return metrics