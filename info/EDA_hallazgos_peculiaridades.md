# Revisión EDA — Peculiaridades del dataset y análisis pendientes

## PARTE 1 — PECULIARIDADES DETECTADAS (ordenadas por severidad)

### 🔴 CRÍTICAS — hay que resolverlas antes de modelar

#### Peculiaridad 1. El dataset está ordenado por `Customer_ID` y la tasa de churn cae al final
- Primeros 10.000 IDs: churn ≈ **49.2%**
- Últimos 10.000 IDs: churn ≈ **37.6%**
- Último quintil (IDs 1.090.001–1.100.000): churn = **36.0 – 39.2%**

**Implicación:** si haces un split secuencial (`train_test_split` sin shuffle), tu test tendrá menos churn que train y las métricas se distorsionan. Incluso con shuffle, probablemente `Customer_ID` codifica una cohorte temporal o un lote de ingesta. **Descarta `Customer_ID` como feature y haz siempre split estratificado + shuffled.**

#### Peculiaridad 2. Clase balanceada artificialmente 50/50
El churn real de un operador de telco es 1–5% mensual. Aquí es **49.56%**. Esto significa que:
- El dataset fue estratificado/resampleado.
- Las probabilidades del modelo **no serán representativas de la realidad**.
- Al reportar a negocio, hay que **recalibrar** con el prior real (p. ej. Platt scaling con un prior declarado) o reportar solo métricas invariantes a la prevalencia (AUC-ROC, AUC-PR).
- No hace falta SMOTE, oversampling ni `class_weight`: ya está balanceado.

#### Peculiaridad 3. Valores negativos imposibles
- `eqpdays` (días con el equipo): **133 valores negativos** (mínimo = −5). Equipo "del futuro" → basura. Limpiar o imputar.
- `rev_Mean`: 5 negativos (ajustes de facturación; defendible).
- `totmrc_Mean`: 23 negativos (descuentos; defendible).

#### Peculiaridad 4. Nulos camuflados como `"U"` o `"Z"` en categóricas
El `isnull()` normal **no los detecta**. Ejemplos:

| Variable      | NaN reales | `"U"`/`"Z"` | % real de faltantes |
|---------------|-----------:|------------:|--------------------:|
| `new_cell`    | 0          | 66.914      | **66.9%**           |
| `marital`     | 1.732      | 37.333      | **39.1%**           |
| `kid0_2`      | 1.732      | 94.256      | **96.0%**           |
| `kid3_5`      | 1.732      | 93.572      | **95.3%**           |
| `kid6_10`     | 1.732      | 90.195      | **91.9%**           |
| `kid11_15`    | 1.732      | 89.454      | **91.2%**           |
| `kid16_17`    | 1.732      | 88.304      | **90.0%**           |
| `prizm_social_one` | 7.388 | 23.613      | **31.0%**           |
| `ethnic`      | 1.732      | 10.945 + 4.425 Z | **17.1%**       |

**Acción:** reemplazar `"U"`/`"Z"` por `NaN` antes del análisis de nulos. Las variables `kid0_2…kid16_17` tienen 90–96% de "U" — tíralas o conviértelas a un único `has_kids` binario.

#### Peculiaridad 5. Multicolinealidad extrema (r > 0.95)
19 pares con |r| > 0.95, incluso **r = 1.000** exacto:

| Par                               | r      |
|-----------------------------------|--------|
| `totcalls` ↔ `adjqty`             | 1.000  |
| `totmou` ↔ `adjmou`               | 1.000  |
| `totrev` ↔ `adjrev`               | 0.998  |
| `plcd_vce_Mean` ↔ `attempt_Mean`  | 0.998  |
| `comp_vce_Mean` ↔ `complete_Mean` | 0.998  |
| `ovrrev_Mean` ↔ `vceovr_Mean`     | 0.995  |
| `ccrndmou_Mean` ↔ `cc_mou_Mean`   | 0.989  |
| `mou_Mean` ↔ `avg3mou`            | 0.981  |

**Acción:** elimina una de cada par antes de entrenar modelos lineales/logit. Para árboles (RF, XGBoost) no es crítico pero ensucia la interpretación de feature importance.

---

### 🟠 IMPORTANTES — afectan calidad pero no rompen el modelo

#### Peculiaridad 6. Variables dummy inútiles (near-zero variance)
Variables con churn rate idéntico entre clases → **cero señal**:
- `truck`: 79.713 en 0, 18.555 en 1 → churn rate ambos ≈ 0.496
- `rv`: idéntico
- `forgntvl`: idéntico

Y sparse (>90% ceros):
- `drop_dat_Mean` (97%), `blck_dat_Mean` (99%), `recv_sms_Mean` (99%), `callfwdv_Mean` (99.6%), `peak_dat_Mean` (91%), `mou_pead_Mean` (91%), `opk_dat_Mean` (90%)

**Acción:** drop o agrúpalas en una flag "usa datos / SMS / callforward".

#### Peculiaridad 7. Alta cardinalidad en categóricas
- `crclscod` (credit class): **54 categorías**, algunas con 61 filas. Muchas con churn rate muy distinto del promedio (ej. `A2`=61.6%, `D4`=37.2%, `E4`=30.6%). Hacer one-hot es inviable (crea 54 columnas muy sparse).

**Acción:** target encoding con smoothing, o agrupar por primera letra (AA/A/A2 → "A"), o por churn rate en buckets.

#### Peculiaridad 8. Outliers extremos
- `uniqsubs = 196` en un cliente (normal: 1–2).
- `mou_Mean` hasta 12.206 minutos/mes (mediana 356) → 34× mediana.
- `rev_Mean` hasta 3.843 USD/mes (mediana 48) → 78× mediana.

**Acción:** winsorizar al p99 o usar `log1p` en ingresos/minutos/cuentas. Para árboles, dejarlos pero reportar en el EDA.

#### Peculiaridad 9. `income` no es el ingreso real
Es una **variable ordinal 1–9** (decil/bucket), no dólares. Trátala como ordinal o categoría, no como continua.

#### Peculiaridad 10. `dwllsize` y `HHstatin` tienen letras con significados ordinales no documentados
A=47k, B=5k, C=1k, J, O, N, D… Sin mapa de traducción, ni target encoding ni ordinal encoding son seguros. En el PDF no viene la descripción → tratarlas como categóricas puras.

---

### 🟡 CONCEPTUALES — cuidado al interpretar

#### Peculiaridad 11. `change_mou` / `change_rev` — ¿leakage?
- Descripción oficial: *"Percentage change in monthly MoU vs previous three month average"*.
- Si el "monthly MoU" es del **mes de observación** y el "previous three months" es el promedio de los **3 meses anteriores**, **NO es leakage**: mide el momentum de uso en el punto de decisión. Es el predictor más fuerte que tienes.
- **Pero** si la ventana de medición del "monthly" se solapa con la ventana del outcome (31–60 días después), sí sería leakage. El PDF no lo aclara 100%.
- **Qué hacer en la presentación:** declararlo como hipótesis, entrenar con y sin la variable, y reportar AUC en ambos casos. Si sin ella el modelo pierde >5 pts de AUC, probablemente hay sobredependencia — sospechoso.

#### Peculiaridad 12. `uniqsubs - actvsubs` podría ser leakage parcial
Si `uniqsubs` > `actvsubs`, significa que hay suscriptores inactivos en el hogar. La correlación con churn es baja (r=0.04), pero para clientes `churn=1` el promedio de la diferencia es 0.22 vs 0.16 (cola larga hasta 143). **Revisar** si la medida de "activo" se toma antes o después de los 31–60 días.

#### Peculiaridad 13. Correlaciones individuales muy bajas
Ninguna variable cruza r = 0.11 con churn. **Esto es normal** en churn (es un fenómeno multifactorial), pero implica que:
- Una regresión logística sin interacciones tendrá AUC modesto (~0.60–0.65).
- Los árboles ganarán claramente (XGBoost/LightGBM AUC típico en estos datasets ~0.68–0.72).
- El verdadero valor del análisis está en **interacciones** entre variables.

---

## PARTE 2 — LO QUE FALTA EN TU EDA

Tu notebook cubre bien la base (shapes, nulos, distribución de target, correlaciones lineales, variables numéricas/categóricas top, change_mou). Te falta:

### Análisis estadístico
1. **Tests de significancia**: Chi-cuadrado para cada categórica vs churn (te dice cuáles discriminan de verdad) y Mann-Whitney U / KS test para numéricas (no asume normalidad; la Pearson que hiciste sí).
2. **VIF (Variance Inflation Factor)** sobre las numéricas: cuantifica la multicolinealidad más allá de pares.
3. **Matriz de correlación completa** (heatmap) en vez de solo las top-20 contra churn — para ver bloques redundantes.
4. **Cramér's V** entre pares de categóricas para detectar redundancia categórica.

### Segmentación / cohortes
5. **Churn por buckets de `months` (antigüedad)**: típicamente curva U o decreciente — clientes en los primeros meses suelen tener más churn, luego se estabiliza.
6. **Churn por buckets de `eqpdays`**: deciles de antigüedad del equipo vs churn rate (te da una curva monótona muy visual).
7. **Segmentación por valor**: buckets de `totrev` (low/mid/high value customer) vs churn.
8. **Matriz `change_mou × eqpdays`**: ¿los clientes con caída fuerte de uso Y equipo viejo tienen churn extremo? Es la interacción que te va a dar la recomendación de negocio más potente.

### Calidad de datos
9. **Análisis de nulos por clase** (MCAR/MAR/MNAR): ¿los clientes que churnean tienen más nulos en demografía? Tu dato: los que churnean tienen 3.54 nulos/fila vs 3.32 de los que no — diferencia pequeña pero sistemática.
10. **Reemplazar "U"/"Z" por NaN** y recalcular el % de nulos real (ver Peculiaridad 4).
11. **Tratamiento de outliers**: decidir y documentar (winsorize p99 vs log vs dejar).
12. **Orden del dataset**: verificar si hay patrón temporal (Peculiaridad 1).

### Variables que no tocaste
13. `months` (antigüedad) — es una variable de negocio clave y apenas la mencionas.
14. `uniqsubs` / `actvsubs` — posible leakage, hay que explorar.
15. `ethnic`, `area`, `crclscod` (más allá de top categorías).
16. `lor` (length of residence), `numbcars`, `adults`, `income` — demografía completa.

### Ingeniería de features para la presentación
17. Crear ratios interpretables: `rev_per_minute = rev_Mean / mou_Mean`, `dropped_rate = drop_vce_Mean / (drop_vce_Mean + comp_vce_Mean)`, `care_intensity = custcare_Mean / mou_Mean`, `change_vs_level = change_mou / avg3mou`.
18. Binarizar `eqpdays > 365` ("equipo antiguo") y cruzar con churn — es un KPI muy legible para stakeholders.

### Visualizaciones que impactan en la presentación
19. **Curva de lift**: "si llamo al top-10% según el modelo, cuántos churners capturo" — **la slide más vendedora**.
20. **Heatmap de interacciones bivariadas** (change_mou bucket × eqpdays bucket, con color = churn rate).
21. **Distribución de churn rate por decil de probabilidad** del modelo — muestra calibración visual.

### Puntos que cambian el modelado
22. **Estrategia de validación robusta**: split estratificado 70/15/15 (train/val/test), no depender solo de CV.
23. **Métricas apropiadas**: dado el 50/50 artificial, reportar AUC-ROC, AUC-PR, F1 a umbral óptimo, y la **curva de ganancia acumulada** (para que negocio decida el umbral).
24. **Prior real**: si puedes, calibra las probabilidades a un prior realista (p. ej. 3%) para reportar "riesgo esperado" de cada cliente.

---

## RESUMEN EJECUTIVO PARA LA PRESENTACIÓN

Las peculiaridades que debes **mencionar explícitamente** (demuestra rigor y sube mucho la nota):

1. **Clase balanceada artificialmente** → decisión consciente de no hacer oversampling, métricas invariantes a la prevalencia, y calibración posterior.
2. **Dataset ordenado por ID con sesgo al final** → split estratificado + shuffled obligatorio.
3. **Nulos camuflados como `"U"`/`"Z"`** → limpieza previa, tratamiento de `kid*` como binaria o drop.
4. **Multicolinealidad masiva** → eliminación de redundantes por VIF o selección automática para lineales.
5. **`change_mou` como predictor estrella pero con asterisco de posible leakage** → entrenar ablation con/sin la variable y mostrar el delta de AUC.
6. **Valores negativos en `eqpdays`** → limpieza documentada.
7. **`income` es ordinal 1–9**, no en dólares → tratamiento correcto.
8. **`crclscod` con 54 categorías** → target encoding con smoothing en vez de one-hot.

Estos 8 puntos son exactamente lo que un panel técnico quiere oír para diferenciarte del candidato promedio que solo corre `df.describe()` y un Random Forest.
