"""
=======================================================================
 Exportador de métricas a Prometheus Pushgateway
=======================================================================

 Envía métricas del pipeline de churn al Pushgateway de Prometheus
 para que sean scrapeadas y visualizadas en Grafana.

 Modo graceful: si el Pushgateway no está disponible, simplemente
 se salta sin afectar la ejecución del pipeline.

 Métricas exportadas:
   - churn_test_auc_roc, churn_test_f1, churn_test_recall, etc.
   - churn_true_positives, churn_false_positives, etc.
   - churn_lift_top_5pct, churn_lift_top_10pct, etc.
   - churn_test_samples, churn_test_churn_rate
   - churn_pipeline_duration_seconds
   - churn_pipeline_runs_total (contador)

 Uso:
   from metrics_exporter import push_metrics_to_prometheus
   push_metrics_to_prometheus(metrics, lift, run_id, duration)
=======================================================================
"""

import os
import time
import urllib.request
import urllib.error


def push_metrics_to_prometheus(metrics: dict, lift: dict, run_id: str,
                                pipeline_duration: float = 0.0,
                                n_samples: int = 0, churn_rate: float = 0.0):
    """
    Envía métricas al Prometheus Pushgateway.

    Args:
        metrics: dict con auc_roc, f1, recall, precision, accuracy, confusion_matrix
        lift: dict con top_5pct, top_10pct, etc.
        run_id: identificador del run
        pipeline_duration: duración total del pipeline en segundos
        n_samples: número de muestras en el test set
        churn_rate: tasa de churn en el test set
    """
    pushgateway_url = os.environ.get('PUSHGATEWAY_URL', '')
    if not pushgateway_url:
        print("[METRICS] ℹ️  PUSHGATEWAY_URL no configurado. Saltando Prometheus.")
        return

    try:
        # Construir payload en formato Prometheus text exposition
        lines = []

        # Métricas principales del modelo
        lines.append(f'# HELP churn_test_auc_roc AUC-ROC del modelo en test set')
        lines.append(f'# TYPE churn_test_auc_roc gauge')
        lines.append(f'churn_test_auc_roc {metrics["auc_roc"]}')

        lines.append(f'# HELP churn_test_f1 F1 score del modelo en test set')
        lines.append(f'# TYPE churn_test_f1 gauge')
        lines.append(f'churn_test_f1 {metrics["f1"]}')

        lines.append(f'# HELP churn_test_recall Recall del modelo en test set')
        lines.append(f'# TYPE churn_test_recall gauge')
        lines.append(f'churn_test_recall {metrics["recall"]}')

        lines.append(f'# HELP churn_test_precision Precision del modelo en test set')
        lines.append(f'# TYPE churn_test_precision gauge')
        lines.append(f'churn_test_precision {metrics["precision"]}')

        lines.append(f'# HELP churn_test_accuracy Accuracy del modelo en test set')
        lines.append(f'# TYPE churn_test_accuracy gauge')
        lines.append(f'churn_test_accuracy {metrics["accuracy"]}')

        # Confusion matrix
        cm = metrics.get('confusion_matrix', {})
        for label, key in [('true_positives', 'TP'), ('true_negatives', 'TN'),
                           ('false_positives', 'FP'), ('false_negatives', 'FN')]:
            lines.append(f'# HELP churn_{label} {label.replace("_", " ").title()} del modelo')
            lines.append(f'# TYPE churn_{label} gauge')
            lines.append(f'churn_{label} {cm.get(key, 0)}')

        # Lift por percentil
        for pct_key, data in lift.items():
            pct = pct_key.replace('top_', '').replace('pct', '')
            lines.append(f'# HELP churn_lift_top_{pct}pct Lift del modelo en top {pct}%')
            lines.append(f'# TYPE churn_lift_top_{pct}pct gauge')
            lines.append(f'churn_lift_top_{pct}pct {data["lift"]}')

        # Metadata del pipeline
        lines.append(f'# HELP churn_test_samples Numero de muestras en test set')
        lines.append(f'# TYPE churn_test_samples gauge')
        lines.append(f'churn_test_samples {n_samples}')

        lines.append(f'# HELP churn_test_churn_rate Tasa de churn en test set')
        lines.append(f'# TYPE churn_test_churn_rate gauge')
        lines.append(f'churn_test_churn_rate {churn_rate}')

        if pipeline_duration > 0:
            lines.append(f'# HELP churn_pipeline_duration_seconds Duracion del pipeline en segundos')
            lines.append(f'# TYPE churn_pipeline_duration_seconds gauge')
            lines.append(f'churn_pipeline_duration_seconds {pipeline_duration:.1f}')

        lines.append(f'# HELP churn_pipeline_runs_total Numero total de ejecuciones')
        lines.append(f'# TYPE churn_pipeline_runs_total counter')
        lines.append(f'churn_pipeline_runs_total 1')

        # Enviar al Pushgateway via HTTP PUT
        payload = '\n'.join(lines) + '\n'
        url = f'{pushgateway_url}/metrics/job/churn_pipeline/run_id/{run_id}'

        req = urllib.request.Request(
            url,
            data=payload.encode('utf-8'),
            method='PUT',
            headers={'Content-Type': 'text/plain; charset=utf-8'}
        )
        urllib.request.urlopen(req, timeout=5)

        print(f"[METRICS] ✅ Métricas enviadas a Prometheus Pushgateway (run: {run_id})")

    except urllib.error.URLError as e:
        print(f"[METRICS] ⚠️  Pushgateway no disponible (no crítico): {e}")
    except Exception as e:
        print(f"[METRICS] ⚠️  Error enviando métricas (no crítico): {e}")
