# Modelo de prediccion de Churn en telecomunicaciones

Una empresa de telecomunicaciones quiere entender por qué sus clientes se dan de baja (churn). 
Te dan un dataset con información de clientes y sus características.

**1. Parte de Data Science (experimentación)**
- Análisis exploratorio de los datos: entender las variables, visualizar patrones, ver qué factores se asocian con el churn.
- Construir un modelo predictivo de churn (el algoritmo que tú elijas).
- Extraer conclusiones de negocio: no solo "qué variables importan", sino recomendaciones accionables tipo "deberíamos hacer X para retener clientes".

**2. Parte de Ingeniería (infraestructura y automatización)**
- Montar Apache Airflow con Docker en local.
- Crear un DAG (pipeline automatizado) con 3 pasos: (1) preparación de datos, (2) entrenamiento del modelo y guardado como pickle, 
    (3) carga del modelo y evaluación con métricas en un conjunto de test.
- Que el pipeline se pueda ejecutar de forma autónoma.

**Extensiones opcionales** (para lucirte más): integrar PostgreSQL para almacenar datos, MLflow para tracking de experimentos, 
    herramientas de monitoreo como Grafana, o cualquier mejora de ingeniería que se te ocurra.

## Resultados

AI_Test_SDG/
├── scripts/
│   ├── data_preparation.py   (12 KB)
│   ├── train_model.py        (7.8 KB)
│   └── evaluate_model.py     (8.0 KB)
├── data/
│   ├── dataset.csv           (input original)
│   └── processed/            (generado por paso 1)
│       ├── X_train.csv, X_val.csv, X_test.csv
│       ├── y_train.csv, y_val.csv, y_test.csv
│       └── preprocessing_artifacts.pkl
└── models/                   (generado por pasos 2 y 3)
    ├── churn_model.pkl
    └── evaluation_report.json

Qué hace cada script:

- data_preparation.py — Carga dataset.csv, aplica las 5 limpiezas de peculiaridades (nulos camuflados, columnas >85% nulas, 
eqpdays negativos, redundancia r>0.98, varianza cero), crea las 8 features nuevas, hace label encoding + imputación con mediana,
split estratificado 70/15/15, y guarda todo. Acepta --input y --output como argumentos.

- train_model.py — Carga los datos procesados, entrena XGBoost (con flag --tune opcional para hacer RandomizedSearchCV), 
evalúa en validation como sanity check, y guarda el modelo como pickle con todos los artefactos incluidos.

- evaluate_model.py — Carga el pickle y el test set, calcula todas las métricas (AUC-ROC, F1, recall, precision, confusion matrix), 
genera la curva de lift completa (5%-50%), compara contra validation, y guarda un evaluation_report.json. 
Incluye un gate de calidad: si AUC < 0.55, retorna error.

Pipeline verificado end-to-end:
data_preparation.py  →  train_model.py  →  evaluate_model.py
       (16s)                (4.5s)              (1s)