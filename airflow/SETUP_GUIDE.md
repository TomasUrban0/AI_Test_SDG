# Guía de Setup — Airflow + Docker para el pipeline de Churn

## Requisitos previos

### 1. Instalar Docker Desktop

- **Windows**: Necesitas Windows 10/11 + WSL2 habilitado.
  - Instalar WSL2: https://learn.microsoft.com/es-es/windows/wsl/install
  - Instalar Docker Desktop: https://www.docker.com/products/docker-desktop/
  - En Docker Desktop → Settings → General → asegurar que "Use WSL2 based engine" esté activado.
  - Asignar al menos **4 GB de RAM** a Docker (Settings → Resources).

- **macOS**: Descargar Docker Desktop desde https://www.docker.com/products/docker-desktop/
  - Asignar al menos **4 GB de RAM** (Settings → Resources).

- **Linux**: Instalar Docker Engine + Docker Compose:
  ```bash
  sudo apt-get update
  sudo apt-get install docker.io docker-compose-v2
  sudo usermod -aG docker $USER
  # Cerrar sesión y volver a abrir
  ```

### 2. Verificar instalación

```bash
docker --version          # Debe mostrar Docker 20.x o superior
docker compose version    # Debe mostrar v2.x
```

---

## Estructura del proyecto

```
AI_Test_SDG/
├── airflow/                      ← TÚ ESTÁS AQUÍ
│   ├── docker-compose.yaml       ← Configuración de los contenedores
│   ├── Dockerfile                ← Imagen custom con xgboost, sklearn, mlflow, etc.
│   ├── requirements.txt          ← Dependencias Python para la imagen
│   ├── .env                      ← AIRFLOW_UID (ajustar en Linux)
│   ├── init-db/
│   │   └── 01_create_churn_db.sql ← Schema de churn_db y mlflow_db (auto-init)
│   ├── dags/
│   │   └── churn_pipeline_dag.py ← El DAG que orquesta el pipeline
│   ├── logs/                     ← Logs de ejecución (auto-generado)
│   ├── plugins/                  ← Plugins de Airflow (vacío)
│   └── config/                   ← Config adicional (vacío)
├── monitoring/                   ← Stack de monitorización
│   ├── prometheus/
│   │   └── prometheus.yml        ← Configuración de scraping (Pushgateway cada 15s)
│   └── grafana/
│       └── provisioning/
│           ├── datasources/
│           │   └── datasource.yml ← Datasource Prometheus (auto-provisioned)
│           └── dashboards/
│               ├── dashboard.yml  ← Proveedor de dashboards
│               └── churn_pipeline.json ← Dashboard "Churn Prediction Pipeline" (6 paneles)
├── scripts/                      ← Los 5 scripts modulares (montados en Docker)
│   ├── data_preparation.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── db_utils.py               ← Utilidades de conexión a PostgreSQL
│   └── metrics_exporter.py       ← Envía métricas al Pushgateway
├── data/                         ← Datos (montados en Docker)
│   ├── dataset.csv               ← Dataset original
│   └── processed/                ← Generado por el pipeline
└── models/                       ← Modelos (montados en Docker)
    ├── churn_model.pkl            ← Generado por el pipeline
    └── evaluation_report.json     ← Generado por el pipeline
```

---

## Paso a paso

### Paso 1: Preparar el dataset

Asegúrate de que `dataset.csv` está en la carpeta `data/`:

```bash
# Desde la raíz del proyecto (AI_Test_SDG/)
mkdir -p data models
cp dataset.csv data/
```

### Paso 2: Ajustar AIRFLOW_UID (solo Linux)

```bash
# En Linux, edita airflow/.env con tu UID:
echo "AIRFLOW_UID=$(id -u)" > airflow/.env
```

En macOS/Windows no hace falta tocarlo (el default 50000 funciona).

### Paso 3: Construir la imagen y levantar Airflow

```bash
cd airflow/

# Primera vez: construir la imagen custom e inicializar la DB
docker compose build
docker compose up airflow-init

# Levantar todos los servicios
docker compose up -d
```

Espera ~30 segundos y verifica que todo esté corriendo:

```bash
docker compose ps
```

Deberías ver 8 contenedores con estado "healthy" o "running":
- `postgres` — Base de datos de Airflow + churn_db + mlflow_db
- `mlflow-server` — MLflow Tracking Server (http://localhost:5000)
- `airflow-webserver` — Interfaz web de Airflow (http://localhost:8080)
- `airflow-scheduler` — Ejecuta los DAGs
- `airflow-triggerer` — Deferrable operators
- `pushgateway` — Recibe métricas push del pipeline (http://localhost:9091)
- `prometheus` — Scrapea Pushgateway cada 15s y almacena series temporales (http://localhost:9090)
- `grafana` — Dashboard "Churn Prediction Pipeline" con 6 paneles (http://localhost:3000)

### Paso 4: Acceder a las interfaces web

- **Airflow:** http://localhost:8080 (usuario: `airflow` / contraseña: `airflow`)
- **MLflow:** http://localhost:5000 (sin autenticación)
- **Grafana:** http://localhost:3000 (usuario: `admin` / contraseña: `admin`)
- **Prometheus:** http://localhost:9090 (sin autenticación)
- **Pushgateway:** http://localhost:9091 (sin autenticación)

### Paso 5: Ejecutar el pipeline

1. En la UI de Airflow, busca el DAG **`churn_prediction_pipeline`**.
2. Actívalo con el toggle de la izquierda (de "paused" a "active").
3. Haz clic en el botón **"Trigger DAG"** (botón de play ▶️).
4. Ve a la pestaña **"Graph"** para ver el progreso en tiempo real.
5. Haz clic en cada task → **"Log"** para ver las métricas impresas.

El pipeline completo tarda aproximadamente **30 segundos**:
- `data_preparation`: ~16s
- `train_model`: ~5s
- `evaluate_model`: ~1s

### Paso 6: Ver los resultados

Después de la ejecución exitosa:

- **Métricas en logs**: Click en `evaluate_model` → Log → verás AUC-ROC, F1, Confusion Matrix, Curva de Lift.
- **Modelo**: `models/churn_model.pkl` (actualizado).
- **Reporte JSON**: `models/evaluation_report.json`.
- **Datos procesados**: `data/processed/` (X_train, X_test, etc.).
- **PostgreSQL (churn_db)**: Métricas, predicciones y features persistidas en 3 tablas:
  - `model_runs` — registro histórico de cada run con métricas y hiperparámetros.
  - `model_predictions` — 15K predicciones por run (probabilidad + predicción + valor real).
  - `processed_features` — features procesadas por split (train/val/test).
- **MLflow**: Experimento `churn_prediction` en http://localhost:5000 con:
  - Run de entrenamiento: 41 hiperparámetros + 4 métricas de validación.
  - Run de evaluación: métricas de test (AUC, F1, recall, precision, accuracy) + confusion matrix + lift.

### Consultar datos en PostgreSQL

```bash
# Conectar a churn_db desde el contenedor de postgres
docker compose exec postgres psql -U airflow -d churn_db

# Ver métricas del último run
SELECT run_id, auc_roc, f1_score, recall, precision_val FROM model_runs ORDER BY run_date DESC LIMIT 5;

# Ver predicciones de mayor riesgo
SELECT * FROM latest_predictions LIMIT 10;

# Contar registros
SELECT count(*) FROM model_predictions;
```

---

## Comandos útiles

```bash
# Ver logs de un servicio específico
docker compose logs airflow-scheduler

# Ejecutar el DAG desde la línea de comandos (sin UI)
docker compose run airflow-cli dags trigger churn_prediction_pipeline

# Listar DAGs disponibles
docker compose run airflow-cli dags list

# Detener todo
docker compose down

# Detener y eliminar volúmenes (reset completo)
docker compose down --volumes --remove-orphans

# Reconstruir la imagen (si cambias requirements.txt)
docker compose build --no-cache
docker compose up -d
```

---

## Solución de problemas comunes

### "Permission denied" en logs/
```bash
# Linux: dar permisos a la carpeta
chmod -R 777 logs/
```

### "No module named 'xgboost'"
La imagen custom no se construyó correctamente:
```bash
docker compose build --no-cache
docker compose up -d
```

### El DAG no aparece en la UI
- Espera ~30 segundos (el scheduler escanea periódicamente).
- Revisa errores de sintaxis: `docker compose run airflow-cli dags list-import-errors`

### "Port 8080 already in use"
Otro servicio usa el puerto 8080:
```bash
# Cambiar el puerto en docker-compose.yaml (línea ports)
# De "8080:8080" a "8081:8080" y acceder en http://localhost:8081
```

### Docker es muy lento / se queda sin memoria
- Asigna más RAM a Docker Desktop (mínimo 4 GB, recomendado 6 GB).
- En Docker Desktop → Settings → Resources → Memory.
