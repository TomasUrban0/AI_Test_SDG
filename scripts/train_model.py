"""
=======================================================================
 PASO 2 — Entrenamiento del modelo de churn
=======================================================================

 Qué hace este script:
   1. Carga los datasets procesados (X_train, y_train, X_val, y_val).
   2. Entrena un XGBoost con los hiperparámetros optimizados.
   3. Opcionalmente ejecuta RandomizedSearchCV para tuning.
   4. Evalúa en el validation set como sanity check.
   5. Guarda el modelo completo como pickle (modelo + artefactos + métricas).

 Inputs:
   - data/processed/X_train.csv, y_train.csv
   - data/processed/X_val.csv, y_val.csv
   - data/processed/preprocessing_artifacts.pkl

 Outputs:
   - models/churn_model.pkl

 Uso:
   python train_model.py --data ../data/processed/ --output ../models/ [--tune]
=======================================================================
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier


RANDOM_STATE = 42


# =====================================================================
# Hiperparámetros
# =====================================================================

# Hiperparámetros por defecto (punto de partida conservador)
DEFAULT_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 1,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'tree_method': 'hist',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': 0,
}

# Espacio de búsqueda para RandomizedSearchCV
TUNING_SPACE = {
    'max_depth': [4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'n_estimators': [200, 300, 500],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [3, 5, 7, 10],
    'reg_alpha': [0, 0.01, 0.1, 0.5],
    'reg_lambda': [0.5, 1.0, 2.0, 5.0],
}


def load_data(data_dir: str):
    """Carga los datasets procesados."""
    print(f"[TRAIN] Cargando datos desde: {data_dir}")

    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))['churn'].values
    X_val = pd.read_csv(os.path.join(data_dir, 'X_val.csv'))
    y_val = pd.read_csv(os.path.join(data_dir, 'y_val.csv'))['churn'].values

    with open(os.path.join(data_dir, 'preprocessing_artifacts.pkl'), 'rb') as f:
        artifacts = pickle.load(f)

    print(f"[TRAIN] Train: {X_train.shape}  |  Val: {X_val.shape}")
    return X_train, y_train, X_val, y_val, artifacts


def train_default(X_train, y_train, X_val, y_val):
    """Entrena XGBoost con hiperparámetros por defecto."""
    print("[TRAIN] Entrenando XGBoost con hiperparámetros por defecto...")
    t0 = time.time()

    model = XGBClassifier(**DEFAULT_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    elapsed = time.time() - t0
    print(f"[TRAIN] Entrenamiento completado en {elapsed:.1f}s")
    return model


def train_with_tuning(X_train, y_train, n_iter=30):
    """Entrena XGBoost con RandomizedSearchCV."""
    print(f"[TRAIN] Ejecutando RandomizedSearchCV ({n_iter} iteraciones x 3 folds)...")
    t0 = time.time()

    base_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        tree_method='hist',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=TUNING_SPACE,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    elapsed = time.time() - t0
    print(f"[TRAIN] Tuning completado en {elapsed:.1f}s")
    print(f"[TRAIN] Mejor AUC-ROC (CV): {search.best_score_:.4f}")
    print(f"[TRAIN] Mejores hiperparámetros:")
    for k, v in sorted(search.best_params_.items()):
        print(f"  {k:25s} = {v}")

    return search.best_estimator_


def evaluate_on_validation(model, X_val, y_val) -> dict:
    """Evalúa el modelo en el validation set (sanity check)."""
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        'val_auc_roc': round(roc_auc_score(y_val, y_proba), 4),
        'val_f1': round(f1_score(y_val, y_pred), 4),
        'val_recall': round(recall_score(y_val, y_pred), 4),
        'val_precision': round(precision_score(y_val, y_pred), 4),
    }

    print(f"\n[TRAIN] Evaluación en VALIDATION SET (sanity check):")
    for k, v in metrics.items():
        print(f"  {k:20s} = {v}")

    return metrics


def save_model(model, artifacts, val_metrics, output_dir, X_train_shape):
    """Guarda el modelo completo como pickle."""
    os.makedirs(output_dir, exist_ok=True)

    model_package = {
        # Modelo
        'model': model,
        'model_name': 'XGBoost',

        # Preprocesamiento (del paso anterior)
        'imputer': artifacts['imputer'],
        'label_encoders': artifacts['label_encoders'],
        'scaler': artifacts['scaler'],
        'feature_names': artifacts['feature_names'],
        'cat_cols': artifacts['cat_cols'],
        'num_cols': artifacts['num_cols'],

        # Métricas de validación
        'val_metrics': val_metrics,

        # Hiperparámetros
        'best_params': model.get_params(),

        # Metadata
        'training_date': pd.Timestamp.now().isoformat(),
        'train_size': X_train_shape[0],
        'n_features': X_train_shape[1],
    }

    pkl_path = os.path.join(output_dir, 'churn_model.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(model_package, f)

    size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
    print(f"\n[TRAIN] Modelo guardado en: {pkl_path} ({size_mb:.2f} MB)")
    return pkl_path


# =====================================================================
# Main
# =====================================================================
def main(data_dir: str, output_dir: str, do_tuning: bool = False, n_iter: int = 30):
    print("=" * 70)
    print(" PASO 2 — ENTRENAMIENTO DEL MODELO")
    print("=" * 70)

    # 1. Carga
    X_train, y_train, X_val, y_val, artifacts = load_data(data_dir)

    # 2. Entrenamiento
    if do_tuning:
        model = train_with_tuning(X_train, y_train, n_iter=n_iter)
    else:
        model = train_default(X_train, y_train, X_val, y_val)

    # 3. Evaluación en validation
    val_metrics = evaluate_on_validation(model, X_val, y_val)

    # 4. Guardar
    save_model(model, artifacts, val_metrics, output_dir, X_train.shape)

    print("\n[TRAIN] ✅ Entrenamiento completado exitosamente.")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenamiento del modelo de churn')
    parser.add_argument('--data', type=str, default='../data/processed/',
                        help='Carpeta con los datasets procesados')
    parser.add_argument('--output', type=str, default='../models/',
                        help='Carpeta donde guardar el modelo')
    parser.add_argument('--tune', action='store_true',
                        help='Activar RandomizedSearchCV (más lento pero busca mejores hiperparámetros)')
    parser.add_argument('--n-iter', type=int, default=30,
                        help='Número de iteraciones para el tuning (default: 30)')
    args = parser.parse_args()

    sys.exit(main(args.data, args.output, do_tuning=args.tune, n_iter=args.n_iter))
