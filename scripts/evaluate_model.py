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
      - Curva de Lift (top-5%, 10%, 15%, 20%, 25%, 30%, 40%, 50%)
   5. Imprime todo en logs (para que Airflow lo capture).
   6. Opcionalmente guarda un reporte JSON con las métricas.
   7. [Extensión] Persiste métricas y predicciones en PostgreSQL (churn_db).

 Inputs:
   - models/churn_model.pkl
   - data/processed/X_test.csv, y_test.csv

 Outputs:
   - Métricas impresas en stdout/logs
   - (Opcional) models/evaluation_report.json
   - [PostgreSQL] Tablas model_runs y model_predictions en churn_db

 Uso:
   python evaluate_model.py --model ../models/churn_model.pkl --data ../data/processed/
=======================================================================
"""

import argparse
import json
import os
import pickle
import sys
import traceback

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
def persist_to_database(run_id, metrics, lift, y_proba, y_pred, y_test,
                        model_name, hyperparameters, n_rows, n_features,
                        churn_rate, model_path):
    """
    [Extensión 3.6] Persiste métricas y predicciones en PostgreSQL.
    Modo graceful: si la DB no está disponible, simplemente se salta.
    """
    try:
        from db_utils import get_engine, save_run_metrics, save_predictions_to_db
        engine = get_engine()
        if engine is not None:
            # Guardar métricas del run
            ok_metrics = save_run_metrics(
                engine, run_id, metrics, lift,
                model_name=model_name,
                hyperparameters=hyperparameters,
                n_rows=n_rows,
                n_features=n_features,
                churn_rate=churn_rate,
                model_path=model_path,
            )
            # Guardar predicciones por cliente
            ok_preds = save_predictions_to_db(
                engine, run_id, y_proba, y_pred, y_actual=y_test
            )
            if ok_metrics and ok_preds:
                print(f"[EVAL] ✅ Métricas y predicciones guardadas en PostgreSQL (run: {run_id})")
            else:
                print("[EVAL] ⚠️  Guardado parcial en PostgreSQL.")
        else:
            print("[EVAL] ℹ️  PostgreSQL no disponible. Solo se guarda JSON.")
    except ImportError:
        print("[EVAL] ℹ️  db_utils no disponible. Solo se guarda JSON.")
    except Exception as e:
        print(f"[EVAL] ⚠️  Error de DB (no crítico): {e}")


def track_test_metrics_mlflow(metrics, lift, model_path):
    """
    [Extensión 3.6 — Fase 2] Loguea métricas de test en MLflow.
    Busca el run activo del experimento churn_prediction (creado por train_model)
    y le añade las métricas de evaluación en test.
    Modo graceful: si MLflow no está disponible, simplemente se salta.
    """
    try:
        import mlflow

        tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', '')
        if not tracking_uri:
            print("[EVAL] ℹ️  MLFLOW_TRACKING_URI no configurado. Saltando MLflow.")
            return

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("churn_prediction")

        run_id_env = os.environ.get('CHURN_RUN_ID', '')

        with mlflow.start_run(run_name=f"{run_id_env}_eval" if run_id_env else None):
            # Métricas principales de test
            mlflow.log_metric("test_auc_roc", metrics['auc_roc'])
            mlflow.log_metric("test_f1", metrics['f1'])
            mlflow.log_metric("test_recall", metrics['recall'])
            mlflow.log_metric("test_precision", metrics['precision'])
            mlflow.log_metric("test_accuracy", metrics['accuracy'])

            # Confusion matrix
            cm = metrics.get('confusion_matrix', {})
            mlflow.log_metric("test_true_negatives", cm.get('TN', 0))
            mlflow.log_metric("test_false_positives", cm.get('FP', 0))
            mlflow.log_metric("test_false_negatives", cm.get('FN', 0))
            mlflow.log_metric("test_true_positives", cm.get('TP', 0))

            # Lift top percentiles
            for key, data in lift.items():
                pct = key.replace('top_', '').replace('pct', '')
                mlflow.log_metric(f"lift_top_{pct}pct", data['lift'])
                mlflow.log_metric(f"gain_top_{pct}pct", data['cumulative_gain'])

            # Log evaluation report JSON como artefacto
            report_path = os.path.join(os.path.dirname(model_path), 'evaluation_report.json')
            if os.path.exists(report_path):
                try:
                    mlflow.log_artifact(report_path, "evaluation")
                except Exception as e_art:
                    print(f"[EVAL] ⚠️  No se pudo loguear artefacto: {e_art}")

            print(f"[EVAL] ✅ Métricas de test registradas en MLflow (run: {mlflow.active_run().info.run_id[:8]}...)")

    except ImportError:
        print("[EVAL] ℹ️  mlflow no instalado. Saltando tracking.")
    except Exception as e:
        print(f"[EVAL] ⚠️  Error de MLflow (no crítico): {e}")
        traceback.print_exc()


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
    y_pred = (y_proba >= 0.5).astype(int)

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

    # 7. [Extensión] Persistir en PostgreSQL
    run_id = os.environ.get('CHURN_RUN_ID', '')
    if not run_id:
        run_id = f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    persist_to_database(
        run_id=run_id,
        metrics=metrics,
        lift=lift,
        y_proba=y_proba,
        y_pred=y_pred,
        y_test=y_test,
        model_name=package.get('model_name', 'XGBoost'),
        hyperparameters=package.get('best_params', {}),
        n_rows=X_test.shape[0],
        n_features=X_test.shape[1],
        churn_rate=float(y_test.mean()),
        model_path=model_path,
    )

    # 8. [Extensión] Loguear métricas de test en MLflow
    track_test_metrics_mlflow(metrics, lift, model_path)

    # 9. [Extensión] Exportar métricas a Prometheus Pushgateway
    try:
        from metrics_exporter import push_metrics_to_prometheus
        push_metrics_to_prometheus(
            metrics=metrics,
            lift=lift,
            run_id=run_id,
            n_samples=X_test.shape[0],
            churn_rate=float(y_test.mean()),
        )
    except ImportError:
        print("[EVAL] ℹ️  metrics_exporter no disponible. Saltando Prometheus.")
    except Exception as e:
        print(f"[EVAL] ⚠️  Error de Prometheus (no crítico): {e}")

    # 10. Verificar que el modelo es mínimamente funcional
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
