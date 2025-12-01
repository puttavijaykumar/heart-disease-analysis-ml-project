from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
import pandas as pd

def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        # Macro / Weighted metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro'
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )

        results[name] = {
            "Accuracy": acc,

            "Precision_Macro": precision_macro,
            "Recall_Macro": recall_macro,
            "F1_Macro": f1_macro,

            "Precision_Weighted": precision_weighted,
            "Recall_Weighted": recall_weighted,
            "F1_Weighted": f1_weighted,

            "Precision_Per_Class": precision_per_class,
            "Recall_Per_Class": recall_per_class,
            "F1_Per_Class": f1_per_class
        }

    return results
