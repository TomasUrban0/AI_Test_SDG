"""
=======================================================================
 PASO 1 — Preparación de datos para el modelo de churn
=======================================================================

 Qué hace este script:
   1. Carga el dataset original (dataset.csv).
   2. Limpia las trampas detectadas en el EDA:
      - Reemplaza nulos camuflados ("U", "Z") por NaN.
      - Elimina columnas con >85% de nulos.
      - Corrige valores negativos en eqpdays.
      - Elimina variables redundantes (r > 0.98).
      - Elimina variables con varianza casi nula.
   3. Aplica feature engineering (8 features nuevas).
   4. Codifica categóricas (LabelEncoder) e imputa nulos (mediana).
   5. Hace un split estratificado 70/15/15 (train/val/test).
   6. Guarda los artefactos en la carpeta de salida (CSV + pickle).
   7. [Extensión] Persiste las features procesadas en PostgreSQL (churn_db).

 Outputs:
   - X_train.csv, y_train.csv
   - X_val.csv, y_val.csv
   - X_test.csv, y_test.csv
   - preprocessing_artifacts.pkl  (encoders, imputer, scaler, etc.)
   - [PostgreSQL] Tabla processed_features en churn_db

 Uso:
   python data_preparation.py --input ../data/dataset.csv --output ../data/processed/
=======================================================================
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# =====================================================================
# Configuración
# =====================================================================
RANDOM_STATE = 42
NULL_THRESHOLD = 0.85       # Columnas con >85% nulos → drop
CORR_THRESHOLD = 0.98       # Pares con |r| > 0.98 → drop uno
NEAR_CONST_THRESHOLD = 0.99 # >99% mismo valor → drop

# Columnas kid (>90% "U") — siempre las eliminamos
KID_COLS = ['kid0_2', 'kid3_5', 'kid6_10', 'kid11_15', 'kid16_17']


def load_data(path: str) -> pd.DataFrame:
    """Carga el CSV con separador ';' y decimal ','."""
    print(f"[PREP] Cargando datos desde: {path}")
    df = pd.read_csv(path, sep=';', decimal=',', low_memory=False)
    print(f"[PREP] Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    return df


def clean_hidden_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza 'U' y 'Z' por NaN en columnas categóricas."""
    cat_cols = df.select_dtypes(include=['object']).columns
    count = 0
    for c in cat_cols:
        mask = df[c].isin(['U', 'Z'])
        n = mask.sum()
        if n > 0:
            df.loc[mask, c] = np.nan
            count += n
    print(f"[PREP] Nulos camuflados reemplazados: {count:,} valores 'U'/'Z' → NaN")
    return df


def drop_high_null_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas con más de NULL_THRESHOLD % de nulos."""
    null_pct = df.isnull().mean()
    to_drop = null_pct[null_pct > NULL_THRESHOLD].index.tolist()
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"[PREP] Columnas eliminadas por exceso de nulos (>{NULL_THRESHOLD*100:.0f}%): {to_drop}")
    return df


def fix_negative_eqpdays(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte eqpdays negativos a NaN."""
    if 'eqpdays' in df.columns:
        n = (df['eqpdays'] < 0).sum()
        if n > 0:
            df.loc[df['eqpdays'] < 0, 'eqpdays'] = np.nan
            print(f"[PREP] eqpdays: {n} valores negativos → NaN")
    return df


def drop_redundant_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina una variable de cada par con correlación > CORR_THRESHOLD."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['churn', 'Customer_ID']]

    corr = df[num_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Correlación de cada variable con churn (para decidir cuál quitar)
    churn_corr = df[num_cols].corrwith(df['churn']).abs()

    to_drop = set()
    for col in upper.columns:
        for row in upper.index:
            v = upper.loc[row, col]
            if pd.notna(v) and v > CORR_THRESHOLD:
                if row not in to_drop and col not in to_drop:
                    loser = row if churn_corr.get(row, 0) < churn_corr.get(col, 0) else col
                    to_drop.add(loser)

    if to_drop:
        df = df.drop(columns=list(to_drop))
        print(f"[PREP] Columnas eliminadas por redundancia (r>{CORR_THRESHOLD}): {len(to_drop)} → {sorted(to_drop)}")
    return df


def drop_near_constant(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas con >99% del mismo valor O dummies sin señal."""
    to_drop = []
    for c in df.columns:
        if c in ['churn', 'Customer_ID']:
            continue
        vc = df[c].value_counts(normalize=True, dropna=False)
        if len(vc) == 0:
            continue
        # Near-constant
        if vc.iloc[0] > NEAR_CONST_THRESHOLD:
            to_drop.append(c)
            continue
        # Dummy sin señal: misma churn rate en 0 y 1
        vals = df[c].dropna().unique()
        if set(vals).issubset({0, 1, 0.0, 1.0}) and len(vals) == 2:
            c0 = df[df[c] == 0]['churn'].mean()
            c1 = df[df[c] == 1]['churn'].mean()
            if abs(c0 - c1) < 0.01:
                to_drop.append(c)

    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"[PREP] Columnas eliminadas por varianza nula/sin señal: {len(to_drop)} → {sorted(to_drop)}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Crea 8 features nuevas interpretables para el negocio."""
    eps = 1e-6

    def first_available(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    # 1) Ingreso por minuto
    df['rev_per_minute'] = df['rev_Mean'] / (df['mou_Mean'] + eps)

    # 2) Tasa de fallos de llamada
    attempts_col = first_available(['attempt_Mean', 'plcd_vce_Mean', 'complete_Mean', 'comp_vce_Mean'])
    fail = df['drop_vce_Mean'].fillna(0) + df['blck_vce_Mean'].fillna(0)
    df['call_fail_rate'] = fail / (df[attempts_col].fillna(0) + eps) if attempts_col else 0

    # 3) Intensidad de contacto con soporte
    df['care_intensity'] = df['custcare_Mean'] / (df['mou_Mean'] + eps)

    # 4) Equipo antiguo (flag binaria)
    df['old_device'] = (df['eqpdays'] > 365).astype('Int64')

    # 5) Caída de uso severa
    df['usage_drop_severe'] = (df['change_mou'] < -50).astype('Int64')

    # 6) Ratio facturación vs media histórica
    df['rev_vs_avg3'] = df['rev_Mean'] / (df['avg3rev'] + eps)

    # 7) Cliente inmaduro
    df['new_customer'] = (df['months'] <= 6).astype('Int64')

    # 8) Engagement score
    comp_col = first_available(['comp_vce_Mean', 'complete_Mean'])
    recv_col = first_available(['recv_vce_Mean', 'iwylis_vce_Mean'])
    df['engagement_score'] = (
        df['mou_Mean'].fillna(0).rank(pct=True) +
        df[comp_col].fillna(0).rank(pct=True) +
        df[recv_col].fillna(0).rank(pct=True)
    ) / 3

    new_feats = ['rev_per_minute', 'call_fail_rate', 'care_intensity',
                 'old_device', 'usage_drop_severe', 'rev_vs_avg3',
                 'new_customer', 'engagement_score']
    print(f"[PREP] Features nuevas creadas: {new_feats}")
    return df


def encode_and_impute(X: pd.DataFrame):
    """Label-encode categóricas, imputa numéricas con mediana, escala para LR."""
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Label Encoding
    label_encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = X[c].fillna('MISSING')
        X[c] = le.fit_transform(X[c])
        label_encoders[c] = le

    # Imputación numérica con mediana
    imputer = SimpleImputer(strategy='median')
    X[num_cols] = imputer.fit_transform(X[num_cols])

    # Scaler (para Logistic Regression baseline)
    scaler = StandardScaler()
    scaler.fit(X)  # Fit on all data, will be used on train only in practice

    feature_names = X.columns.tolist()

    artifacts = {
        'label_encoders': label_encoders,
        'imputer': imputer,
        'scaler': scaler,
        'feature_names': feature_names,
        'cat_cols': cat_cols,
        'num_cols': num_cols,
    }

    print(f"[PREP] Encoding: {len(cat_cols)} categóricas, {len(num_cols)} numéricas")
    print(f"[PREP] Nulos restantes: {X.isnull().sum().sum()}")
    return X, artifacts


def split_data(X: pd.DataFrame, y: np.ndarray):
    """Split estratificado 70/15/15."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, shuffle=True, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176,  # 0.176 de 85% ≈ 15% del total
        stratify=y_temp, shuffle=True, random_state=RANDOM_STATE
    )
    print(f"[PREP] Split:")
    print(f"  Train:      {X_train.shape[0]:>6,} ({y_train.mean():.4f} churn)")
    print(f"  Validation: {X_val.shape[0]:>6,} ({y_val.mean():.4f} churn)")
    print(f"  Test:       {X_test.shape[0]:>6,} ({y_test.mean():.4f} churn)")
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_outputs(output_dir, X_train, X_val, X_test, y_train, y_val, y_test, artifacts):
    """Guarda CSVs y pickle de artefactos."""
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_val.to_csv(os.path.join(output_dir, 'X_val.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    pd.Series(y_train).to_csv(os.path.join(output_dir, 'y_train.csv'), index=False, header=['churn'])
    pd.Series(y_val).to_csv(os.path.join(output_dir, 'y_val.csv'), index=False, header=['churn'])
    pd.Series(y_test).to_csv(os.path.join(output_dir, 'y_test.csv'), index=False, header=['churn'])

    with open(os.path.join(output_dir, 'preprocessing_artifacts.pkl'), 'wb') as f:
        pickle.dump(artifacts, f)

    print(f"[PREP] Archivos guardados en: {output_dir}")
    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        size = os.path.getsize(fpath) / 1024
        print(f"  {fname:35s} {size:>8.1f} KB")


# =====================================================================
# Main
# =====================================================================
def persist_to_database(run_id, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    [Extensión 3.6] Persiste las features procesadas en PostgreSQL.
    Modo graceful: si la DB no está disponible, simplemente se salta.
    """
    try:
        from db_utils import get_engine, save_features_to_db
        engine = get_engine()
        if engine is not None:
            ok = save_features_to_db(
                engine, run_id,
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
            )
            if ok:
                print(f"[PREP] ✅ Features guardadas en PostgreSQL (run: {run_id})")
            else:
                print("[PREP] ⚠️  No se pudieron guardar las features en PostgreSQL.")
        else:
            print("[PREP] ℹ️  PostgreSQL no disponible. Continuando solo con CSVs.")
    except ImportError:
        print("[PREP] ℹ️  db_utils no disponible. Continuando solo con CSVs.")
    except Exception as e:
        print(f"[PREP] ⚠️  Error de DB (no crítico): {e}")


def main(input_path: str, output_dir: str):
    print("=" * 70)
    print(" PASO 1 — PREPARACIÓN DE DATOS")
    print("=" * 70)

    # 1. Carga
    df = load_data(input_path)

    # 2. Limpieza de trampas
    df = clean_hidden_nulls(df)
    df = drop_high_null_cols(df)
    df = fix_negative_eqpdays(df)
    df = drop_redundant_cols(df)
    df = drop_near_constant(df)

    # 3. Feature engineering
    df = feature_engineering(df)

    # 4. Separar target y features
    y = df['churn'].values
    X = df.drop(columns=['churn', 'Customer_ID'], errors='ignore')

    # Eliminar columnas auxiliares que podrían quedar
    aux = ['months_bucket', 'eqpdays_bucket', 'totrev_bucket']
    X = X.drop(columns=[c for c in aux if c in X.columns])

    # 5. Encoding + imputación
    X, artifacts = encode_and_impute(X)

    # 6. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 7. Guardar CSVs (siempre — es el fallback principal)
    save_outputs(output_dir, X_train, X_val, X_test, y_train, y_val, y_test, artifacts)

    # 8. [Extensión] Persistir en PostgreSQL
    run_id = os.environ.get('CHURN_RUN_ID', '')
    if not run_id:
        from datetime import datetime
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    persist_to_database(run_id, X_train, y_train, X_val, y_val, X_test, y_test)

    print("\n[PREP] ✅ Preparación de datos completada exitosamente.")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preparación de datos para el modelo de churn')
    parser.add_argument('--input', type=str, default='../data/dataset.csv',
                        help='Ruta al dataset original (dataset.csv)')
    parser.add_argument('--output', type=str, default='../data/processed/',
                        help='Carpeta donde guardar los datasets procesados')
    args = parser.parse_args()

    sys.exit(main(args.input, args.output))
