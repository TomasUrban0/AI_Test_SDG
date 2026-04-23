-- =============================================================================
-- Init script para PostgreSQL — crea la base de datos churn_db
-- =============================================================================
-- Se ejecuta automáticamente la primera vez que arranca el contenedor de
-- PostgreSQL gracias al volumen montado en /docker-entrypoint-initdb.d/
--
-- Tablas:
--   processed_features  → Features procesadas listas para modelar
--   model_runs          → Registro de cada ejecución del pipeline (MLOps básico)
--   model_predictions   → Predicciones por cliente del último run
-- =============================================================================

-- Crear la base de datos (si no existe)
-- Nota: PostgreSQL no permite IF NOT EXISTS en CREATE DATABASE,
-- pero este script solo se ejecuta la primera vez (init)
CREATE DATABASE churn_db;

-- Conectar a churn_db para crear las tablas
\connect churn_db;

-- -----------------------------------------------------------------
-- Tabla: processed_features
-- Almacena las features procesadas después del paso de preparación.
-- Sustituye a los CSVs (X_train, X_val, X_test) con un campo
-- "split" que indica a qué conjunto pertenece cada fila.
-- -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS processed_features (
    id              SERIAL PRIMARY KEY,
    run_id          VARCHAR(64) NOT NULL,       -- ID del pipeline run
    split           VARCHAR(10) NOT NULL,       -- 'train', 'val', 'test'
    churn           SMALLINT NOT NULL,           -- target variable
    features        JSONB NOT NULL,              -- todas las features como JSON
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_pf_run_id ON processed_features(run_id);
CREATE INDEX idx_pf_split ON processed_features(split);

-- -----------------------------------------------------------------
-- Tabla: model_runs
-- Registro histórico de cada ejecución del pipeline completo.
-- Permite comparar métricas entre runs y detectar degradación.
-- -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_runs (
    id              SERIAL PRIMARY KEY,
    run_id          VARCHAR(64) NOT NULL UNIQUE,
    run_date        TIMESTAMP DEFAULT NOW(),

    -- Datos de entrada
    n_rows          INTEGER,
    n_features      INTEGER,
    churn_rate       FLOAT,

    -- Hiperparámetros del modelo
    model_name      VARCHAR(50),
    hyperparameters JSONB,

    -- Métricas en test set
    auc_roc         FLOAT,
    f1_score        FLOAT,
    recall          FLOAT,
    precision_val   FLOAT,          -- "precision" es palabra reservada
    accuracy        FLOAT,

    -- Confusion matrix
    true_negatives  INTEGER,
    false_positives INTEGER,
    false_negatives INTEGER,
    true_positives  INTEGER,

    -- Lift
    lift_top_5pct   FLOAT,
    lift_top_10pct  FLOAT,
    lift_top_20pct  FLOAT,

    -- Metadata
    model_path      VARCHAR(255),
    status          VARCHAR(20) DEFAULT 'completed'
);

-- -----------------------------------------------------------------
-- Tabla: model_predictions
-- Predicciones por cliente del último run.
-- Útil para alimentar el CRM con scores de riesgo.
-- -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_predictions (
    id              SERIAL PRIMARY KEY,
    run_id          VARCHAR(64) NOT NULL,
    customer_index  INTEGER NOT NULL,           -- índice en el test set
    churn_proba     FLOAT NOT NULL,             -- probabilidad de churn
    churn_pred      SMALLINT NOT NULL,           -- predicción binaria
    churn_actual    SMALLINT,                    -- valor real (si disponible)
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_mp_run_id ON model_predictions(run_id);
CREATE INDEX idx_mp_proba ON model_predictions(churn_proba DESC);

-- -----------------------------------------------------------------
-- Vista: latest_predictions
-- Facilita consultar las predicciones del último run.
-- -----------------------------------------------------------------
CREATE OR REPLACE VIEW latest_predictions AS
SELECT mp.*
FROM model_predictions mp
INNER JOIN (
    SELECT run_id FROM model_runs ORDER BY run_date DESC LIMIT 1
) lr ON mp.run_id = lr.run_id
ORDER BY mp.churn_proba DESC;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO airflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO airflow;
