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
│   ├── Dockerfile                ← Imagen custom con xgboost, sklearn, etc.
│   ├── requirements.txt          ← Dependencias Python para la imagen
│   ├── .env                      ← AIRFLOW_UID (ajustar en Linux)
│   ├── dags/
│   │   └── churn_pipeline_dag.py ← El DAG que orquesta el pipeline
│   ├── logs/                     ← Logs de ejecución (auto-generado)
│   ├── plugins/                  ← Plugins de Airflow (vacío)
│   └── config/                   ← Config adicional (vacío)
├── scripts/                      ← Los 3 scripts modulares (montados en Docker)
│   ├── data_preparation.py
│   ├── train_model.py
│   └── evaluate_model.py
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

Deberías ver 4 contenedores con estado "healthy":
- `postgres`
- `airflow-webserver`
- `airflow-scheduler`
- `airflow-triggerer`

### Paso 4: Acceder a la interfaz web

Abre en tu navegador: **http://localhost:8080**

- **Usuario:** `airflow`
- **Contraseña:** `airflow`

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
