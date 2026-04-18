# =============================================================================
# Dockerfile — Imagen custom de Airflow con dependencias de ML
# =============================================================================
#
# ¿Por qué una imagen custom?
# La imagen base de Airflow no incluye xgboost, scikit-learn, etc.
# Necesitamos instalar las dependencias que usan nuestros scripts.
#
# Build:  docker build -t airflow-churn:latest .
# (docker-compose lo hace automáticamente con "build: .")
# =============================================================================

FROM apache/airflow:2.8.1-python3.10

# Copiar e instalar dependencias
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
