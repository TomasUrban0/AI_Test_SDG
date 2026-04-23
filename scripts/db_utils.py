"""
=======================================================================
 Utilidades de base de datos — Conexión a PostgreSQL (churn_db)
=======================================================================

 Módulo reutilizable para que los scripts del pipeline puedan
 escribir y leer datos de PostgreSQL. Diseñado para ser "graceful":
 si la base de datos no está disponible, el pipeline sigue funcionando
 con CSVs (modo fallback).

 Variables de entorno:
   CHURN_DB_HOST     (default: postgres)
   CHURN_DB_PORT     (default: 5432)
   CHURN_DB_NAME     (default: churn_db)
   CHURN_DB_USER     (default: airflow)
   CHURN_DB_PASSWORD  (default: airflow)

 Uso:
   from db_utils import get_engine, save_features_to_db, save_run_metrics, save_predictions_to_db
=======================================================================
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# Conexión
# =====================================================================

def get_connection_string() -> str:
    """Construye la connection string desde variables de entorno."""
    host = os.environ.get('CHURN_DB_HOST', 'postgres')
    port = os.environ.get('CHURN_DB_PORT', '5432')
    dbname = os.environ.get('CHURN_DB_NAME', 'churn_db')
    user = os.environ.get('CHURN_DB_USER', 'airflow')
    password = os.environ.get('CHURN_DB_PASSWORD', 'airflow')
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"


def get_engine():
    """
    Crea un SQLAlchemy engine para churn_db.
    Retorna None si no se puede conectar (modo graceful).
    """
    try:
        from sqlalchemy import create_engine, text
        conn_str = get_connection_string()
        engine = create_engine(conn_str, pool_pre_ping=True, pool_size=3)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("[DB] Conexión a PostgreSQL (churn_db) establecida.")
        return engine
    except Exception as e:
        logger.warning(f"[DB] No se pudo conectar a PostgreSQL: {e}")
        logger.warning("[DB] Continuando en modo CSV (sin persistencia en DB).")
        return None


def generate_run_id() -> str:
    """Genera un ID único para este run del pipeline."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"run_{ts}_{short_uuid}"


# =====================================================================
# Escritura de features procesadas
# =====================================================================

def save_features_to_db(
    engine,
    run_id: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> bool:
    """
    Guarda las features procesadas en la tabla processed_features.

    En vez de guardar todas las columnas como columnas SQL (que serían ~88),
    usamos JSONB para las features. Esto es más flexible y no requiere
    ALTER TABLE si cambian las features.

    Returns True si se guardó correctamente, False si falló.
    """
    if engine is None:
        return False

    try:
        import json
        import math

        splits = [
            ('train', X_train, y_train),
            ('val', X_val, y_val),
            ('test', X_test, y_test),
        ]

        total_rows = 0
        conn = engine.raw_connection()
        try:
            cur = conn.cursor()
            for split_name, X, y in splits:
                # Preparar batch de valores
                from psycopg2.extras import execute_values
                values = []
                for i in range(len(X)):
                    feat_dict = {}
                    for k, v in X.iloc[i].items():
                        if isinstance(v, (np.integer,)):
                            feat_dict[k] = int(v)
                        elif isinstance(v, (np.floating, float)):
                            f = float(v)
                            feat_dict[k] = None if (math.isnan(f) or math.isinf(f)) else f
                        else:
                            feat_dict[k] = v
                    values.append((
                        run_id, split_name, int(y[i]),
                        json.dumps(feat_dict, default=str),
                    ))

                execute_values(
                    cur,
                    "INSERT INTO processed_features (run_id, split, churn, features) VALUES %s",
                    values,
                    template="(%s, %s, %s, %s::jsonb)",
                    page_size=5000,
                )
                total_rows += len(values)
                logger.info(f"[DB] {split_name}: {len(values):,} filas guardadas")

            conn.commit()
            cur.close()
        finally:
            conn.close()

        logger.info(f"[DB] Total: {total_rows:,} filas en processed_features (run: {run_id})")
        return True

    except Exception as e:
        logger.warning(f"[DB] Error guardando features: {e}")
        return False


# =====================================================================
# Escritura de métricas del modelo
# =====================================================================

def save_run_metrics(
    engine,
    run_id: str,
    metrics: dict,
    lift: dict,
    model_name: str = "XGBoost",
    hyperparameters: dict = None,
    n_rows: int = None,
    n_features: int = None,
    churn_rate: float = None,
    model_path: str = None,
) -> bool:
    """
    Guarda las métricas de un run en la tabla model_runs.
    Returns True si se guardó correctamente.
    """
    if engine is None:
        return False

    try:
        import json
        cm = metrics.get('confusion_matrix', {})

        import math

        def _clean_value(v):
            """Limpia un valor para que sea JSON-safe."""
            if v is None:
                return None
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                f = float(v)
                if math.isnan(f) or math.isinf(f):
                    return None
                return f
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    return None
                return v
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, np.bool_):
                return bool(v)
            return v

        def _clean_dict(d):
            """Limpia recursivamente un dict para JSON válido."""
            if not isinstance(d, dict):
                return _clean_value(d)
            return {k: _clean_dict(v) if isinstance(v, dict) else _clean_value(v)
                    for k, v in d.items()}

        # Preparar hyperparameters como JSON string limpio
        hp_json_str = None
        if hyperparameters:
            hp_clean = _clean_dict(hyperparameters)
            hp_json_str = json.dumps(hp_clean, default=str)

        conn = engine.raw_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO model_runs (
                    run_id, run_date, n_rows, n_features, churn_rate,
                    model_name, hyperparameters,
                    auc_roc, f1_score, recall, precision_val, accuracy,
                    true_negatives, false_positives, false_negatives, true_positives,
                    lift_top_5pct, lift_top_10pct, lift_top_20pct,
                    model_path, status
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s::jsonb,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s
                )
            """, (
                run_id, datetime.now(), n_rows, n_features, churn_rate,
                model_name, hp_json_str,
                metrics.get('auc_roc'), metrics.get('f1'), metrics.get('recall'),
                metrics.get('precision'), metrics.get('accuracy'),
                cm.get('TN'), cm.get('FP'), cm.get('FN'), cm.get('TP'),
                lift.get('top_5pct', {}).get('lift'),
                lift.get('top_10pct', {}).get('lift'),
                lift.get('top_20pct', {}).get('lift'),
                model_path, 'completed',
            ))
            conn.commit()
            cur.close()
        finally:
            conn.close()

        logger.info(f"[DB] Run guardado en model_runs: {run_id} (AUC={metrics.get('auc_roc')})")
        return True

    except Exception as e:
        logger.warning(f"[DB] Error guardando run: {e}")
        return False


# =====================================================================
# Escritura de predicciones por cliente
# =====================================================================

def save_predictions_to_db(
    engine,
    run_id: str,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    y_actual: np.ndarray = None,
) -> bool:
    """
    Guarda las predicciones de cada cliente en model_predictions.
    Útil para alimentar el CRM con scores de riesgo.
    """
    if engine is None:
        return False

    try:
        records = []
        for i in range(len(y_proba)):
            records.append({
                'run_id': run_id,
                'customer_index': i,
                'churn_proba': float(y_proba[i]),
                'churn_pred': int(y_pred[i]),
                'churn_actual': int(y_actual[i]) if y_actual is not None else None,
            })

        df_records = pd.DataFrame(records)
        df_records.to_sql(
            'model_predictions',
            engine,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=5000,
        )

        logger.info(f"[DB] {len(records):,} predicciones guardadas (run: {run_id})")
        return True

    except Exception as e:
        logger.warning(f"[DB] Error guardando predicciones: {e}")
        return False


# =====================================================================
# Lectura — consultas útiles
# =====================================================================

def get_latest_metrics(engine) -> Optional[dict]:
    """Lee las métricas del último run exitoso."""
    if engine is None:
        return None

    try:
        query = """
            SELECT * FROM model_runs
            WHERE status = 'completed'
            ORDER BY run_date DESC
            LIMIT 1
        """
        df = pd.read_sql(query, engine)
        if len(df) == 0:
            return None
        return df.iloc[0].to_dict()
    except Exception as e:
        logger.warning(f"[DB] Error leyendo métricas: {e}")
        return None


def get_metrics_history(engine, limit: int = 20) -> Optional[pd.DataFrame]:
    """Lee el historial de métricas de los últimos N runs."""
    if engine is None:
        return None

    try:
        query = f"""
            SELECT run_id, run_date, auc_roc, f1_score, recall, precision_val,
                   lift_top_10pct, n_rows, model_name
            FROM model_runs
            WHERE status = 'completed'
            ORDER BY run_date DESC
            LIMIT {limit}
        """
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.warning(f"[DB] Error leyendo historial: {e}")
        return None
