"""
=======================================================================
 DAG: churn_pipeline — Pipeline completo de Churn Prediction
=======================================================================

 Este DAG implementa el ciclo de vida de un modelo de ML en 3 pasos:

   1. data_preparation  →  Carga, limpia y prepara los datos
   2. train_model       →  Entrena XGBoost y guarda como pickle
   3. evaluate_model    →  Carga el pickle, evalúa en holdout, logea métricas

 Grafo:  data_preparation >> train_model >> evaluate_model

 Schedule: None (ejecución manual / trigger externo)
           En producción se podría poner '@weekly' o '@monthly'
           para reentrenar periódicamente con datos frescos.

 Notas:
   - Los scripts están en /opt/airflow/scripts/ (montados por docker-compose).
   - Los datos en /opt/airflow/data/ y los modelos en /opt/airflow/models/.
   - Cada task usa PythonOperator para mayor control y logging.
   - Si un paso falla, el DAG se detiene (no ejecuta los siguientes).
=======================================================================
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import os
import sys

# =====================================================================
# Configuración de rutas
# =====================================================================
# Estas rutas son INTERNAS del contenedor Docker (montadas en docker-compose)
SCRIPTS_DIR = os.environ.get('CHURN_SCRIPTS_DIR', '/opt/airflow/scripts')
DATA_DIR = os.environ.get('CHURN_DATA_DIR', '/opt/airflow/data')
MODELS_DIR = os.environ.get('CHURN_MODELS_DIR', '/opt/airflow/models')

# Rutas específicas
RAW_DATA_PATH = os.path.join(DATA_DIR, 'dataset.csv')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_PATH = os.path.join(MODELS_DIR, 'churn_model.pkl')

# Añadir scripts al path para poder importarlos
sys.path.insert(0, SCRIPTS_DIR)


# =====================================================================
# Funciones que ejecuta cada task
# =====================================================================

def _set_shared_run_id(**context):
    """Genera un run_id compartido y lo inyecta como variable de entorno."""
    from datetime import datetime as dt
    run_id = f"run_{dt.now().strftime('%Y%m%d_%H%M%S')}"
    os.environ['CHURN_RUN_ID'] = run_id
    # También lo pasa via XCom para que los tasks downstream lo hereden
    context['ti'].xcom_push(key='churn_run_id', value=run_id)
    print(f"[DAG] Run ID compartido: {run_id}")
    return run_id


def run_data_preparation(**context):
    """
    TASK 1: Preparación de datos.

    Carga dataset.csv, aplica limpieza de peculiaridades (nulos camuflados,
    valores negativos, multicolinealidad, varianza nula), crea features
    nuevas, codifica categóricas, imputa nulos, hace split estratificado
    70/15/15, y guarda los datasets procesados.

    Output: data/processed/ con X_train, X_val, X_test, y_*, artifacts
    """
    from data_preparation import main as prep_main

    # Generar run_id compartido para todo el pipeline
    run_id = _set_shared_run_id(**context)
    print(f"[DAG] CHURN_RUN_ID = {run_id}")

    print("=" * 70)
    print(f" TASK 1 — Preparación de datos")
    print(f" Input:  {RAW_DATA_PATH}")
    print(f" Output: {PROCESSED_DIR}")
    print("=" * 70)

    result = prep_main(
        input_path=RAW_DATA_PATH,
        output_dir=PROCESSED_DIR
    )

    if result != 0:
        raise RuntimeError("La preparación de datos falló con código de salida != 0")

    # Verificar que los archivos se crearon
    expected = ['X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv',
                'preprocessing_artifacts.pkl']
    for f in expected:
        fpath = os.path.join(PROCESSED_DIR, f)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Archivo esperado no encontrado: {fpath}")

    print("[DAG] ✅ Task 1 completada. Archivos verificados.")


def run_train_model(**context):
    """
    TASK 2: Entrenamiento del modelo.

    Carga los datos procesados del paso anterior, entrena un XGBoost
    con hiperparámetros optimizados, evalúa en validation como sanity
    check, y guarda el modelo completo (modelo + artefactos + métricas)
    como pickle.

    Output: models/churn_model.pkl
    """
    from train_model import main as train_main

    # Heredar run_id del task anterior
    run_id = context['ti'].xcom_pull(task_ids='data_preparation', key='churn_run_id')
    if run_id:
        os.environ['CHURN_RUN_ID'] = run_id
        print(f"[DAG] CHURN_RUN_ID = {run_id}")

    print("=" * 70)
    print(f" TASK 2 — Entrenamiento del modelo")
    print(f" Input:  {PROCESSED_DIR}")
    print(f" Output: {MODELS_DIR}")
    print("=" * 70)

    result = train_main(
        data_dir=PROCESSED_DIR,
        output_dir=MODELS_DIR,
        do_tuning=False  # Cambiar a True para activar RandomizedSearchCV
    )

    if result != 0:
        raise RuntimeError("El entrenamiento falló con código de salida != 0")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo no encontrado en: {MODEL_PATH}")

    # Logear tamaño del modelo
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"[DAG] Modelo guardado: {MODEL_PATH} ({size_mb:.2f} MB)")
    print("[DAG] ✅ Task 2 completada. Modelo verificado.")


def run_evaluate_model(**context):
    """
    TASK 3: Evaluación del modelo.

    Carga el pickle del paso anterior y el holdout test set, genera
    predicciones, calcula métricas completas (AUC-ROC, F1, Recall,
    Precision, Confusion Matrix, Curva de Lift), y las imprime en
    los logs de Airflow. También guarda un evaluation_report.json.

    Las métricas quedan visibles en los logs del task en la UI de Airflow.
    """
    from evaluate_model import main as eval_main

    # Heredar run_id del task inicial
    run_id = context['ti'].xcom_pull(task_ids='data_preparation', key='churn_run_id')
    if run_id:
        os.environ['CHURN_RUN_ID'] = run_id
        print(f"[DAG] CHURN_RUN_ID = {run_id}")

    print("=" * 70)
    print(f" TASK 3 — Evaluación del modelo")
    print(f" Modelo: {MODEL_PATH}")
    print(f" Data:   {PROCESSED_DIR}")
    print("=" * 70)

    result = eval_main(
        model_path=MODEL_PATH,
        data_dir=PROCESSED_DIR,
        save_json=True
    )

    if result != 0:
        raise RuntimeError(
            "La evaluación falló. AUC-ROC < 0.55 — el modelo no supera el umbral mínimo de calidad."
        )

    print("[DAG] ✅ Task 3 completada. Métricas logeadas y reporte guardado.")


# =====================================================================
# Definición del DAG
# =====================================================================

default_args = {
    'owner': 'data-science',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='churn_prediction_pipeline',
    default_args=default_args,
    description='Pipeline completo de predicción de churn: preparación → entrenamiento → evaluación',
    schedule_interval=None,  # Ejecución manual. En producción: '@weekly' o '@monthly'
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['churn', 'ml', 'pipeline', 'sdg'],
    doc_md=__doc__,
) as dag:

    # Task 1: Preparación de datos
    task_data_prep = PythonOperator(
        task_id='data_preparation',
        python_callable=run_data_preparation,
        doc_md=run_data_preparation.__doc__,
    )

    # Task 2: Entrenamiento del modelo
    task_train = PythonOperator(
        task_id='train_model',
        python_callable=run_train_model,
        doc_md=run_train_model.__doc__,
    )

    # Task 3: Evaluación del modelo
    task_evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=run_evaluate_model,
        doc_md=run_evaluate_model.__doc__,
    )

    # =====================================================================
    # Grafo de dependencias (DAG)
    #
    #   data_preparation  →  train_model  →  evaluate_model
    #
    # Si data_preparation falla, train_model NO se ejecuta.
    # Si train_model falla, evaluate_model NO se ejecuta.
    # =====================================================================
    task_data_prep >> task_train >> task_evaluate
