"""
=======================================================================
 PASO 3 — Evaluación del modelo sobre el holdout (test set)
=======================================================================

 Qué hace este script:
   1. Carga el modelo entrenado (pickle).
   2. Carga el holdout set (X_test, y_test).
   3. Genera predicciones.
   4. Calcula métricas completas:
      - AUC-ROC, Average Precision, F1, Recall, Precision, Accuracy
      - Confusion Matrix
      - Curva de Lift (top-10%, 20%, 30%)
   5. Imprime todo en logs (para que Airflow lo capture).
   6. Opcionalmente guarda un reporte JSON con las métricas.

 Inputs:
   - models/churn_model.pkl
   - data/processed/X_test.csv, y_test.csv

 Outputs:
   - Métricas impresas en stdout/logs
   - (Opcional) models/evaluation_report.json

 Uso:
   python evaluate_model.py --model ../models/churn_model.pkl --data ../data/processed/
=======================================================================
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_model(model_path: str) -> dict:
    """Carga el paquete del modelo desde pickle."""
    print(f"[EVAL] Cargando modelo desde: {model_path}")
    with open(model_path, 'rb') as f:
        package = pickle.load(f)
    print(f"[EVAL] Modelo cargado: {package['model_name']}")
    print(f"[EVAL] Fecha de entrenamiento: {package['training_date']}")
    print(f"[EVAL] Features: {package['n_features']}")
    return package


def load_test_data(data_dir: str):
    """Carga el holdout set."""
    print(f"[EVAL] Cargando test set desde: {data_dir}")
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))['churn'].values
    print(f"[EVAL] Test set: {X_test.shape[0]:,} filas x {X_test.shape[1]} columnas")
    print(f"[EVAL] Churn rate en test: {y_test.mean():.4f}")
    return X_test, y_test


def compute_metrics(y_true, y_proba, threshold=0.5) -> dict:
    """Calcula todas las métricas de evaluación."""
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        'threshold': threshold,
        'auc_roc': round(roc_auc_score(y_true, y_proba), 4),
        'avg_precision': round(average_precision_score(y_true, y_proba), 4),
        'f1': round(f1_score(y_true, y_pred), 4),
        'recall': round(recall_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred), 4),
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = {
        'TN': int(cm[0, 0]),
        'FP': int(cm[0, 1]),
        'FN': int(cm[1, 0]),
        'TP': int(cm[1, 1]),
    }

    return metrics


def compute_lift(y_true, y_proba) -> dict:
    """Calcula la curva de lift para puntos clave."""
    order = np.argsort(y_proba)[::-1]
    y_sorted = y_true[order]

    n = len(y_sorted)
    total_positives = y_sorted.sum()
    cum_positives = np.cumsum(y_sorted)

    lift_results = {}
    for pct in [5, 10, 15, 20, 25, 30, 40, 50]:
        idx = int(pct / 100 * n) - 1
        if idx < 0:
            idx = 0
        cum_gain = cum_positives[idx] / total_positives
        lift = cum_gain / (pct / 100)
        lift_results[f'top_{pct}pct'] = {
            'cumulative_gain': round(cum_gain, 4),
            'lift': round(lift, 2),
            'churners_captured': int(cum_positives[idx]),
            'total_churners': int(total_positives),
            'customers_contacted': int(idx + 1),
        }

    return lift_results


def print_report(metrics: dict, lift: dict, val_metrics: dict = None):
    """Imprime el reporte completo en logs."""
    print("\n" + "=" * 70)
    print(" EVALUACIÓN FINAL — HOLDOUT TEST SET")
    print("=" * 70)

    print(f"\n  AUC-ROC:           {metrics['auc_roc']}")
    print(f"  Average Precision: {metrics['avg_precision']}")
    print(f"  F1 Score:          {metrics['f1']}")
    print(f"  Recall:            {metrics['recall']}")
    print(f"  Precision:         {metrics['precision']}")
    print(f"  Accuracy:          {metrics['accuracy']}")
    print(f"  Threshold:         {metrics['threshold']}")

    cm = metrics['confusion_matrix']
    print(f"\n  Confusion Matrix:")
    print(f"                    Pred No Churn    Pred Churn")
    print(f"    Real No Churn     {cm['TN']:>7,}        {cm['FP']:>7,}")
    print(f"    Real Churn        {cm['FN']:>7,}        {cm['TP']:>7,}")

    if val_metrics:
        print(f"\n  Comparación con Validation Set:")
        print(f"    AUC-ROC:  Val={val_metrics.get('val_auc_roc', 'N/A')}  |  Test={metrics['auc_roc']}  "
              f"|  Delta={metrics['auc_roc'] - val_metrics.get('val_auc_roc', 0):+.4f}")
        print(f"    F1:       Val={val_metrics.get('val_f1', 'N/A')}  |  Test={metrics['f1']}  "
              f"|  Delta={metrics['f1'] - val_metrics.get('val_f1', 0):+.4f}")

    print(f"\n  Curva de Lift (valor para negocio):")
    print(f"  {'% Contactados':>15}  {'% Churners capturados':>22}  {'Lift':>6}  {'Churners':>10}")
    print(f"  {'-'*15}  {'-'*22}  {'-'*6}  {'-'*10}")
    for key in sorted(lift.keys(), key=lambda x: int(x.split('_')[1].replace('pct', ''))):
        data = lift[key]
        pct = key.split('_')[1].replace('pct', '')
        print(f"  {pct + '%':>15}  {data['cumulative_gain']*100:>21.1f}%  {data['lift']:>5.2f}x  "
              f"{data['churners_captured']:>5}/{data['total_churners']}")

    print("\n" + "=" * 70)


def save_report(metrics: dict, lift: dict, output_path: str):
    """Guarda el reporte como JSON."""
    report = {
        'metrics': metrics,
        'lift': lift,
        'timestamp': pd.Timestamp.now().isoformat(),
    }
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[EVAL] Reporte guardado en: {output_path}")


# =====================================================================
# Main
# =====================================================================
def main(model_path: str, data_dir: str, save_json: bool = True):
    print("=" * 70)
    print(" PASO 3 — EVALUACIÓN DEL MODELO")
    print("=" * 70)

    # 1. Cargar modelo
    package = load_model(model_path)
    model = package['model']

    # 2. Cargar test set
    X_test, y_test = load_test_data(data_dir)

    # 3. Predecir
    print("[EVAL] Generando predicciones...")
    y_proba = model.predict_proba(X_test)[:, 1]

    # 4. Calcular métricas
    metrics = compute_metrics(y_test, y_proba, threshold=0.5)
    lift = compute_lift(y_test, y_proba)

    # 5. Imprimir reporte en logs
    val_metrics = package.get('val_metrics', None)
    print_report(metrics, lift, val_metrics)

    # 6. (Opcional) Guardar JSON
    if save_json:
        report_dir = os.path.dirname(model_path)
        report_path = os.path.join(report_dir, 'evaluation_report.json')
        save_report(metrics, lift, report_path)

    # 7. Verificar que el modelo es mínimamente funcional
    if metrics['auc_roc'] < 0.55:
        print("\n[EVAL] ⚠️  ALERTA: AUC-ROC < 0.55 — el modelo apenas supera el azar.")
        print("[EVAL] Revisar el preprocesamiento y el entrenamiento.")
        return 1

    print("\n[EVAL] ✅ Evaluación completada exitosamente.")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluación del modelo de churn')
    parser.add_argument('--model', type=str, default='../models/churn_model.pkl',
                        help='Ruta al pickle del modelo')
    parser.add_argument('--data', type=str, default='../data/processed/',
                        help='Carpeta con X_test.csv y y_test.csv')
    parser.add_argument('--no-json', action='store_true',
                        help='No guardar reporte JSON')
    args = parser.parse_args()

    sys.exit(main(args.model, args.data, save_json=not args.no_json))
