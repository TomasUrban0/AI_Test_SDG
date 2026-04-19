# Guia de Estudio Completa — Prueba Tecnica Churn Prediction

**Objetivo:** Prepararte para defender todas las decisiones tecnicas de la prueba tecnica de SDG Group en una sesion de 30+ minutos con preguntas.

---

## 1. CHURN EN TELECOMUNICACIONES — Contexto de Negocio

### 1.1 Que es el churn

El churn (o tasa de abandono) es el porcentaje de clientes que dejan de usar un servicio en un periodo determinado. En telecomunicaciones, se mide generalmente de forma mensual.

Tipos de churn:
- **Voluntario**: el cliente decide irse (insatisfaccion, mejor oferta de la competencia, cambio de necesidades).
- **Involuntario**: la empresa cancela al cliente (impago, fraude).
- **Contractual**: el cliente no renueva al final del contrato.

### 1.2 Por que importa economicamente

El coste de adquisicion de un nuevo cliente (CAC) en telecomunicaciones es de 5 a 7 veces mayor que el coste de retencion. Esto se debe a los subsidios de equipos, campanas de marketing, comisiones de venta, y el tiempo hasta que el cliente es rentable.

Ejemplo numerico: Si una telco tiene 10 millones de clientes con un ARPU (Average Revenue Per User) de 30€/mes y un churn mensual del 2%, pierde 200.000 clientes al mes = 6 millones €/mes en ingresos. Reducir el churn del 2% al 1.8% (0.2 puntos porcentuales) retendria 20.000 clientes/mes = 600.000 €/mes adicionales = 7.2 millones €/año.

### 1.3 Como se aborda con ML

El enfoque clasico es la clasificacion binaria: para cada cliente, predecir si hara churn (1) o no (0) en los proximos N dias/meses. Las predicciones se usan para generar listas ordenadas por riesgo, que el equipo de retencion usa para priorizar intervenciones (llamadas, descuentos, upgrades).

La metrica clave no es accuracy sino **la capacidad de priorizar** (lift, AUC-ROC): no necesitamos predecir perfectamente, sino identificar a los de mayor riesgo antes que a los de menor riesgo.

---

## 2. EXPLORATORY DATA ANALYSIS (EDA)

### 2.1 Por que hacer EDA

El EDA no es un paso formal que se hace "porque toca". Es la fase donde se entiende la naturaleza de los datos, se detectan anomalias, y se forman hipotesis. Un modelo entrenado sin EDA previo puede aprender patrones espureos (como IDs correlacionados con el target).

### 2.2 Que buscar en un EDA para churn

**Distribuciones del target:** ¿Esta balanceado? Un dataset real de churn tendra un 1-5% de positivos. Un dataset balanceado al 50/50 es artificial.

**Valores faltantes:** No solo isnull(). Buscar valores centinela como "U", "Z", "-1", "999", "Unknown", "N/A" que se usan como placeholder. Estos son "nulos camuflados" que un analisis automatico no detectaria.

**Outliers:** Valores negativos donde no deberia haberlos (ej: dias de antiguedad negativos), valores extremos que podrian ser errores de medicion o fraude.

**Correlaciones:** Pares de variables con correlacion perfecta (r=1.0) indican informacion duplicada. Mantener ambas no añade señal y puede causar problemas en modelos lineales (multicolinealidad).

**Varianza:** Variables donde >99% de las filas tienen el mismo valor no aportan informacion discriminativa. Eliminarlas reduce ruido.

**Relacion con el target:** Tests estadisticos para verificar que las variables realmente discriminan entre churn y no-churn:
- **Chi-cuadrado** para categoricas vs target.
- **Mann-Whitney U** para numericas vs target (no asume normalidad).
- **Correlacion de Spearman** como alternativa robusta a Pearson.

### 2.3 Las trampas de nuestro dataset

Nuestro dataset de 100K x 100 tenia las siguientes trampas intencionadas:

| Trampa | Tipo | Severidad | Solucion |
|--------|------|-----------|----------|
| Customer_ID ordenado y correlacionado con churn | Data leakage | CRITICA | Eliminar del modelo |
| Balance artificial 50/50 | Diseño experimental | MEDIA | Tener en cuenta al interpretar accuracy |
| Nulos camuflados ("U", "Z") | Calidad de datos | ALTA | Reemplazar por NaN |
| Valores negativos en eqpdays | Error de datos | ALTA | Convertir a NaN, imputar |
| Multicolinealidad (r=1.000) | Redundancia | ALTA | Eliminar una del par |
| Varianza casi nula (>99% mismo valor) | Ruido | MEDIA | Eliminar |
| Dummies sin señal (churn rate identico en 0 y 1) | Ruido | MEDIA | Eliminar |
| Columnas con >85% nulos | Datos insuficientes | ALTA | Eliminar |
| change_mou (posible leakage) | Sospecha de leakage | CRITICA | Estudio de ablacion → validada como OK |

### 2.4 Data leakage — concepto critico

Data leakage ocurre cuando informacion que no estaria disponible en el momento de la prediccion se incluye en los features. Tipos:

- **Target leakage**: una variable que es consecuencia del target, no causa. Ej: "fecha de cancelacion" usada para predecir churn.
- **Train-test contamination**: usar informacion del test set durante el entrenamiento. Ej: normalizar ANTES de hacer el split.
- **Feature leakage temporal**: usar datos del futuro. Ej: "ventas del proximo mes" para predecir churn este mes.

En nuestro caso, Customer_ID ordenado era leakage evidente. change_mou era sospechoso, pero el estudio de ablacion (delta AUC < 5pp al eliminarlo) confirmo que es una señal legitima.

**Como hacer un estudio de ablacion:** Entrenas dos modelos identicos, uno con la variable sospechosa y otro sin ella. Si el delta en la metrica principal es grande (>5-10 puntos), es probable leakage. Si es pequeño (<5 puntos), la variable es legitima.

---

## 3. FEATURE ENGINEERING

### 3.1 Por que crear features nuevas

Los modelos de ML aprenden relaciones entre features y target. Si una relacion es compleja (ratio de dos variables, interaccion de tres), el modelo puede tener dificultades para descubrirla por si solo. Feature engineering explicita estas relaciones.

### 3.2 Principios de buen feature engineering

- **Interpretabilidad**: cada feature debe tener un significado de negocio claro. "rev_per_minute" es comprensible; "PCA_component_7" no lo es.
- **Derivacion de dominio**: las mejores features vienen del conocimiento del dominio, no de transformaciones matematicas genericas.
- **No crear leakage**: nunca usar informacion del futuro ni del target para crear features.
- **Documentar**: cada feature debe tener su logica de calculo documentada.

### 3.3 Nuestras 8 features y su logica de negocio

**rev_per_minute = rev_Mean / mou_Mean**: Cuanto paga por minuto. Un cliente que paga mucho por poco uso puede sentir que el servicio es caro. Un valor muy alto indica plan desproporcionado.

**call_fail_rate = (drop_vce + blck_vce) / intentos**: Calidad de servicio percibida. Un 10% de fallos es frustrante. Esta feature captura la experiencia del usuario con la red.

**care_intensity = custcare_Mean / mou_Mean**: Ratio soporte/uso. Un cliente que llama mucho al soporte pero usa poco el servicio esta insatisfecho. Es un proxy de frustracion.

**old_device = (eqpdays > 365)**: Equipo con mas de un año. En telecomunicaciones, los equipos viejos suelen estar asociados a contratos vencidos o falta de compromiso.

**usage_drop_severe = (change_mou < -50)**: Caida de uso dramatica. Una reduccion de mas de 50 minutos respecto al periodo anterior es una señal de desenganche activo.

**rev_vs_avg3 = rev_Mean / avg3rev**: Cambio en facturacion. Si el ingreso actual es mucho menor que la media de los ultimos 3 meses, puede indicar downgrade o reduccion de uso.

**new_customer = (months <= 6)**: Cliente inmaduro. Los primeros 6 meses son criticos: el cliente aun no ha formado habito ni tiene switching costs altos.

**engagement_score = rank(mou + comp_vce + recv_vce) / 3**: Score compuesto. Combina tres dimensiones de engagement: cuanto habla, cuantas llamadas completa, y cuantas recibe. Un cliente con alto engagement en las tres dimensiones es de bajo riesgo.

---

## 4. PREPROCESAMIENTO

### 4.1 LabelEncoder vs OneHotEncoder

**LabelEncoder** convierte cada categoria a un numero entero (A=0, B=1, C=2). Es simple y no aumenta la dimensionalidad.

**OneHotEncoder** crea una columna binaria por cada categoria. Con una variable de 50 categorias, crea 50 columnas nuevas.

Para **arboles de decision** (XGBoost, Random Forest), LabelEncoder funciona igual de bien porque el arbol puede hacer splits en cualquier valor ("categoria <= 2" captura las mismas particiones). La ventaja es que no explota la dimensionalidad.

Para **modelos lineales** (Logistic Regression, SVM), OneHotEncoder es necesario porque un encoding ordinal implica una relacion de orden que puede no existir.

Nosotros usamos LabelEncoder porque nuestro modelo principal es XGBoost.

### 4.2 Imputacion de nulos

**Mediana**: Robusta a outliers. Si una variable de ingresos tiene un rango de 0-500 pero unos pocos outliers en 10.000, la mediana no se ve afectada.

**Media**: Sensible a outliers. En el ejemplo anterior, la media estaria inflada.

**Moda**: Para categoricas.

**KNN Imputer**: Imputa basandose en los K vecinos mas cercanos. Mas sofisticado pero mas lento.

**Regla practica**: Si una columna tiene >85% de nulos, mejor eliminarla. Imputar con cualquier metodo seria inventar datos.

### 4.3 Split estratificado 70/15/15

- **70% Train**: Para entrenar el modelo.
- **15% Validation**: Para tuning de hiperparametros, early stopping, y sanity check. Se puede usar multiples veces sin contaminar el test.
- **15% Test (holdout)**: Solo se usa UNA VEZ, al final, para la evaluacion definitiva. Nunca se toca durante el desarrollo.

**Estratificado** significa que cada particion mantiene la misma proporcion de churn/no-churn que el dataset original. Sin estratificar, una particion podria tener 55% churn y otra 45%, lo cual sesga las metricas.

El 0.176 que aparece en el codigo se calcula asi: queremos 15% del total como validation. Tras extraer 15% como test, nos queda 85%. 15/85 ≈ 0.176, que es el porcentaje del 85% restante que corresponde a validation.

---

## 5. MODELOS DE MACHINE LEARNING

### 5.1 Logistic Regression (baseline)

**Que es**: Un modelo lineal que predice la probabilidad de clase positiva usando la funcion sigmoide: P(y=1|x) = 1 / (1 + exp(-w'x)).

**Ventajas**: Muy rapido, interpretable (cada coeficiente indica el efecto de la variable), funciona bien con pocas features.

**Desventajas**: Solo captura relaciones lineales. No maneja interacciones automaticamente.

**AUC en nuestro caso**: 0.626. Es el piso que cualquier modelo mas complejo debe superar.

**Por que se incluye**: Como baseline interpretable. Si un XGBoost complejo solo mejora 1 punto de AUC sobre LR, la complejidad adicional no se justifica. En nuestro caso, la diferencia es de 7.6 puntos (0.702 vs 0.626), lo cual si justifica XGBoost.

### 5.2 Decision Tree

**Que es**: Particiones recursivas del espacio de features. En cada nodo, elige la feature y el punto de corte que maximiza la pureza (Gini o entropia).

**AUC**: 0.638.

**Problema**: Overfitting. Un arbol profundo memoriza el train set. Poda (pruning) ayuda pero pierde capacidad.

### 5.3 Random Forest

**Que es**: Ensemble de muchos decision trees entrenados con muestras bootstrap (bagging) y subconjuntos aleatorios de features. La prediccion es el voto/promedio de todos los arboles.

**AUC**: 0.671.

**Ventaja sobre un arbol**: Reduce varianza gracias a la decorrelacion entre arboles.

**Desventaja vs boosting**: No optimiza secuencialmente; cada arbol es independiente.

### 5.4 XGBoost (modelo elegido)

**Que es**: eXtreme Gradient Boosting. Construye arboles secuencialmente, donde cada nuevo arbol corrige los errores del anterior. Optimiza una funcion de perdida mediante gradient descent en el espacio de funciones.

**AUC**: 0.694 (antes de tuning) → 0.7023 (despues de tuning en test).

**Hiperparametros clave y que hacen**:

| Parametro | Que controla | Rango tipico |
|-----------|-------------|--------------|
| n_estimators | Numero de arboles | 100-1000 |
| max_depth | Profundidad maxima de cada arbol | 3-8 |
| learning_rate | Cuanto "pesa" cada arbol nuevo | 0.01-0.3 |
| subsample | Fraccion de filas usadas por arbol | 0.6-1.0 |
| colsample_bytree | Fraccion de features usadas por arbol | 0.6-1.0 |
| min_child_weight | Peso minimo en un nodo hoja | 1-10 |
| reg_alpha (L1) | Regularizacion que lleva coeficientes a 0 (sparsity) | 0-1 |
| reg_lambda (L2) | Regularizacion que reduce coeficientes (smooth) | 0.5-5 |
| scale_pos_weight | Peso de la clase positiva (para desbalance) | 1 (si balanceado) |

**Diferencia entre L1 (alpha) y L2 (lambda)**:
- L1 (Lasso) añade |w| a la perdida → tiende a hacer coeficientes exactamente 0 → seleccion de features implicita.
- L2 (Ridge) añade w² a la perdida → reduce coeficientes pero no los elimina → suaviza el modelo.
- Usar ambos (Elastic Net) combina las ventajas.

**Por que XGBoost sobre LightGBM**:
Ambos empatan en AUC (0.694). Elegimos XGBoost porque:
1. Regularizacion L1/L2 nativa mas controlable.
2. SHAP TreeExplainer esta optimizado especificamente para XGBoost.
3. Ecosistema mas maduro (mas documentacion, mas respuestas en StackOverflow).
4. scale_pos_weight para manejar desbalance en produccion.

LightGBM es mas rapido (leaf-wise vs level-wise) pero en un dataset de 100K esa diferencia es irrelevante.

### 5.5 LightGBM (comparativa)

**Que es**: Light Gradient Boosting Machine. Similar a XGBoost pero usa un crecimiento leaf-wise (escoge la hoja con mayor reduccion de perdida) en vez de level-wise. Esto lo hace mas rapido pero potencialmente mas propenso a overfitting en datasets pequeños.

**AUC**: 0.694 (empata con XGBoost).

**Ventajas**: Mas rapido en datasets grandes (>1M filas), soporte nativo para categoricas.

### 5.6 Por que NO deep learning

Para datos tabulares de tamaño moderado (<1M filas), los gradient boosting trees superan consistentemente a las redes neuronales. Esto esta documentado en:
- Grinsztajn et al. (2022): "Why do tree-based models still outperform deep learning on tabular data?"
- Shwartz-Ziv & Armon (2022): "Tabular data: Deep learning is not all you need"

Ademas, SHAP TreeExplainer calcula valores exactos para arboles, mientras que para redes neuronales usa aproximaciones (DeepExplainer, GradientExplainer) que son menos fiables.

---

## 6. HYPERPARAMETER TUNING

### 6.1 GridSearch vs RandomSearch vs Bayesian

**GridSearch**: Prueba todas las combinaciones de una rejilla. Si tienes 5 parametros con 4 valores cada uno: 4^5 = 1024 combinaciones x 3 folds = 3072 entrenamientos. Intratable.

**RandomizedSearch**: Muestrea N combinaciones aleatorias del espacio. Bergstra & Bengio (2012) demostraron que con el mismo presupuesto computacional, random search encuentra mejores hiperparametros porque no desperdicia evaluaciones en regiones irrelevantes.

**Bayesian (Optuna, Hyperopt)**: Usa un modelo probabilistico para elegir la siguiente combinacion a probar basandose en los resultados anteriores. Mas eficiente que random pero mas complejo de implementar.

Nosotros usamos **RandomizedSearch con 30 iteraciones** porque es un buen equilibrio entre eficiencia y simplicidad. 30 iteraciones con 3-fold = 90 entrenamientos, que con 100K filas se completa en minutos.

### 6.2 Cross-validation

**K-fold cross-validation** divide el train set en K particiones. Se entrena K veces, cada vez usando K-1 particiones para entrenar y 1 para validar. La metrica final es el promedio de las K evaluaciones.

**Estratificado** mantiene la proporcion del target en cada fold.

Usamos **3-fold** porque con 100K filas, cada fold tiene ~67K filas, que es suficiente para una estimacion estable. La diferencia entre 3-fold y 5-fold es minima en precision pero 5-fold tarda un 67% mas.

### 6.3 Metrica de optimizacion: AUC-ROC

Optimizamos AUC-ROC porque:
1. Es **threshold-independent**: no nos compromete a un punto de corte durante el tuning.
2. Mide la **capacidad de ranking**: cuanto de bien ordena el modelo a los clientes de mayor a menor riesgo.
3. Es **robusto a desbalance** (comparado con accuracy).

Alternativas que se podrian haber usado:
- **Average Precision (AP)**: mas informativa cuando hay mucho desbalance. En nuestro caso (50/50) no aporta ventaja.
- **F1 Score**: pero requiere fijar un threshold, lo cual no queremos durante tuning.
- **Log Loss**: castiga predicciones calibradas. Util si nos importa que P(churn)=0.7 realmente signifique 70%.

---

## 7. METRICAS DE EVALUACION

### 7.1 Confusion Matrix

|  | Pred: No Churn | Pred: Churn |
|--|---------------|-------------|
| **Real: No Churn** | TN (True Negative) = 4,718 | FP (False Positive) = 2,782 |
| **Real: Churn** | FN (False Negative) = 2,644 | TP (True Positive) = 4,856 |

- **TP**: Churners correctamente identificados → recibirán campana de retención.
- **TN**: No-churners correctamente ignorados → no gastamos en ellos.
- **FP**: No-churners marcados como churn → recibirán campana innecesaria (coste de retencion sin beneficio).
- **FN**: Churners no detectados → se pierden (coste de oportunidad perdida).

### 7.2 AUC-ROC (0.7023)

La curva ROC grafica TPR (Recall) vs FPR (1-Especificidad) para todos los thresholds posibles de 0 a 1.

- AUC = 1.0: modelo perfecto (separa perfectamente las clases).
- AUC = 0.5: modelo aleatorio (no discrimina).
- AUC = 0.7: capacidad discriminativa moderada-buena.

**Interpretacion intuitiva**: Si tomas un cliente que realmente hara churn y uno que no, hay un 70.23% de probabilidad de que el modelo asigne mayor probabilidad de churn al primero.

### 7.3 Precision (0.6420)

**Formula**: TP / (TP + FP) = 4,856 / (4,856 + 2,782) = 0.6420

**En español**: De cada 100 clientes que el modelo dice "va a hacer churn", 64 realmente lo hacen.

**Implicacion de negocio**: Si envias una oferta de retencion de 50€ a cada cliente marcado, el 36% del gasto es "desperdiciado" en clientes que no iban a irse. Pero ojo: esos clientes pueden percibir positivamente la oferta, asi que no todo es perdida.

### 7.4 Recall / Sensibilidad (0.6476)

**Formula**: TP / (TP + FN) = 4,856 / (4,856 + 2,644) = 0.6476

**En español**: De cada 100 clientes que realmente van a hacer churn, identificamos 65. Los otros 35 se nos escapan.

**Implicacion de negocio**: Cada churner no detectado es un ingreso perdido. Si el ARPU es 30€/mes y el lifetime value es de 3 años, un churner perdido vale ~1,080€.

### 7.5 F1-Score (0.6448)

**Formula**: 2 × (Precision × Recall) / (Precision + Recall) = 2 × (0.642 × 0.648) / (0.642 + 0.648) = 0.6448

La **media armonica** penaliza valores bajos mas que la media aritmetica. Si precision fuera 0.9 y recall 0.1, la media seria 0.5 pero el F1 seria 0.18. Esto refleja mejor que un modelo tan desequilibrado no es util.

### 7.6 Threshold: como ajustarlo

El modelo produce probabilidades continuas (0 a 1). El threshold determina desde que probabilidad clasificamos como churn:

- **Threshold alto (0.7)**: Solo los muy seguros se marcan como churn. Alta precision, bajo recall. Pocos falsos positivos, pero muchos churners escapan.
- **Threshold bajo (0.3)**: Marcamos a muchos como churn. Alto recall, baja precision. Capturamos mas churners, pero contactamos a muchos que no lo son.

**La eleccion depende del coste relativo**:
- Si el coste de retencion es bajo y el ARPU es alto → threshold bajo (priorizar recall).
- Si el coste de retencion es alto y el ARPU es bajo → threshold alto (priorizar precision).

### 7.7 Curva de Lift

La lift curve mide cuantas veces mejor es nuestro modelo respecto a seleccionar al azar.

**Formula**: Lift@k% = (% churners capturados en top k%) / k%

Nuestros resultados:
- Top 5%: Lift 1.72x (72% mejor que azar)
- Top 10%: Lift 1.62x
- Top 20%: Lift 1.50x
- Top 30%: Lift 1.40x
- Top 50%: Lift 1.25x

**Interpretacion para negocio**: "Si solo podemos llamar al 10% de nuestros clientes, ¿a cuales llamamos?" Con el modelo, contactando al top 10% capturamos un 62% mas de churners que contactando al azar.

---

## 8. SHAP (SHapley Additive exPlanations)

### 8.1 Fundamento teorico

SHAP esta basado en los valores de Shapley de la teoria de juegos cooperativos (Nobel 2012 a Lloyd Shapley). En el contexto de ML, cada feature es un "jugador" y la prediccion es el "premio" a repartir.

El valor SHAP de una feature para una prediccion es su contribucion marginal promediada sobre todas las coaliciones posibles de features.

**Propiedades axiomaticas**:
- **Eficiencia**: la suma de todos los SHAP values iguala la prediccion menos la prediccion base.
- **Simetria**: features con la misma contribucion tienen el mismo SHAP value.
- **Dummy**: una feature que no contribuye tiene SHAP value 0.
- **Aditividad**: para modelos ensemble, los SHAP values son la suma de los de cada modelo.

### 8.2 TreeExplainer

Para modelos de arboles (XGBoost, LightGBM, Random Forest), existe **TreeExplainer** que calcula SHAP values **exactos** en tiempo polinomico (O(TLD²) donde T=arboles, L=hojas, D=profundidad). Esto es clave: para otros modelos, SHAP usa aproximaciones.

### 8.3 Tipos de graficos SHAP

- **Summary plot (beeswarm)**: Cada punto es una instancia. El color indica el valor de la feature (rojo=alto, azul=bajo). La posicion horizontal indica el SHAP value (derecha=aumenta churn, izquierda=disminuye). Da una vision global de la importancia y la direccion del efecto.

- **Bar plot**: Media del |SHAP value| por feature. Da la importancia global sin la direccion.

- **Dependence plot**: SHAP value de una feature vs su valor real. Muestra relaciones no lineales. Ej: SHAP de change_mou es muy negativo para valores positivos (uso aumenta → menos churn) y positivo para valores negativos (uso decrece → mas churn).

- **Waterfall plot**: Para un cliente individual, muestra como cada feature contribuye a su prediccion especifica. Ej: "Este cliente tiene riesgo 78% porque change_mou=-80 (+0.12), eqpdays=720 (+0.08), months=3 (+0.05)..."

### 8.4 Nuestros top predictores y por que

1. **change_mou**: El cliente que reduce drasticamente su uso esta en proceso de abandonar. Es la señal mas temprana y mas fuerte.
2. **mou_Mean**: Poco uso absoluto = poco engagement = bajo switching cost.
3. **eqpdays**: Equipo viejo = contrato posiblemente vencido = libertad para irse.
4. **months**: Cliente nuevo = no ha formado habito = volatil.
5. **rev_Mean**: Baja facturacion = bajo valor percibido del servicio.

### 8.5 SHAP vs Feature Importance nativa

La feature importance de XGBoost se basa en "gain" (cuanto reduce la perdida cada feature al usarla en un split). Problemas:
- Es **inconsistente**: añadir una feature irrelevante puede cambiar la importancia de las demas.
- No distingue **direccion**: no sabes si un valor alto de la feature aumenta o disminuye la prediccion.
- Varias metricas disponibles (gain, weight, cover) que dan resultados diferentes.

SHAP resuelve todos estos problemas gracias a sus propiedades axiomaticas.

---

## 9. APACHE AIRFLOW

### 9.1 Que es Airflow

Apache Airflow es un orquestador de workflows. Permite definir, programar y monitorizar pipelines de datos como grafos dirigidos aciclicos (DAGs). Desarrollado en Airbnb (2014), es open-source desde 2016 y se graduó como proyecto Apache de primer nivel.

### 9.2 Conceptos clave

**DAG (Directed Acyclic Graph)**: Define el workflow completo. Es un grafo donde los nodos son tasks y las aristas son dependencias. "Aciclico" significa que no hay ciclos (una task no puede depender de si misma directa o indirectamente).

**Task**: Una unidad de trabajo. Se define con un Operator.

**Operator**: La logica de lo que hace una task:
- **PythonOperator**: Ejecuta una funcion Python. El que usamos nosotros porque da maximo control y logging.
- **BashOperator**: Ejecuta un comando bash.
- **DockerOperator**: Ejecuta un contenedor Docker.
- **Otros**: EmailOperator, S3Operator, BigQueryOperator, etc.

**Executor**: Determina como se ejecutan las tasks:
- **SequentialExecutor**: Una task a la vez. Solo para desarrollo.
- **LocalExecutor**: Multiples tasks en paralelo usando procesos locales. **El que usamos nosotros.**
- **CeleryExecutor**: Distribuye tasks a workers via Redis/RabbitMQ. Para produccion a escala.
- **KubernetesExecutor**: Cada task en un pod de Kubernetes. Maximo aislamiento.

**Schedule**: Cuando se ejecuta el DAG:
- `None`: solo manual/trigger externo. **Lo que usamos.**
- `'@daily'`, `'@weekly'`, `'@monthly'`: periodicidad predefinida.
- Cron expression: `'0 2 * * 1'` = cada lunes a las 2 AM.

### 9.3 Nuestro DAG

```
data_preparation >> train_model >> evaluate_model
```

Tres PythonOperator en serie. Si `data_preparation` falla, `train_model` no se ejecuta. Si `train_model` falla, `evaluate_model` no se ejecuta.

Cada task importa la funcion `main()` del script correspondiente. Los scripts son los mismos que se pueden ejecutar independientemente desde la linea de comandos (son modulos con `if __name__ == '__main__'`).

**default_args**:
- `retries: 1` y `retry_delay: 2min`: si una task falla, se reintenta una vez tras 2 minutos.
- `depends_on_past: False`: cada ejecucion es independiente de las anteriores.
- `email_on_failure: False`: no enviamos emails (no hay SMTP configurado).

### 9.4 Por que PythonOperator y no BashOperator

- **Mejor manejo de errores**: Python exceptions vs exit codes de bash.
- **Logging integrado**: print() se captura automaticamente en los logs de Airflow.
- **Paso de contexto**: podemos usar XCom para pasar datos entre tasks (aunque en nuestro caso no lo necesitamos porque las tasks se comunican via archivos).
- **Testing**: podemos testear las funciones de Python directamente con pytest.

### 9.5 Por que LocalExecutor

Para un solo DAG con 3 tasks secuenciales, CeleryExecutor seria sobredimensionado:
- Celery requiere Redis o RabbitMQ como message broker: +1 servicio, +500MB RAM.
- Celery Workers: +1 o mas servicios, +1GB RAM cada uno.
- Total extra: ~2GB RAM sin beneficio, porque no tenemos tasks paralelas.

LocalExecutor usa procesos del sistema operativo, funciona con PostgreSQL (que ya tenemos para la metadata), y es suficiente para nuestro caso.

---

## 10. DOCKER Y DOCKER COMPOSE

### 10.1 Por que Docker

Docker encapsula la aplicacion y todas sus dependencias en un contenedor aislado. Esto garantiza que "si funciona en mi maquina, funciona en todas". Para nuestro caso:
- La version de Python (3.10), Airflow (2.8.1), XGBoost (2.1.4), etc., quedan fijadas.
- No hay conflictos con las dependencias del sistema del usuario.
- Replicable en cualquier maquina con Docker instalado.

### 10.2 Nuestro Dockerfile

```dockerfile
FROM apache/airflow:2.8.1-python3.10
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
```

Hereda de la imagen oficial de Airflow (que ya incluye el webserver, scheduler, y todas las dependencias de Airflow) y añade nuestras dependencias de ML (xgboost, scikit-learn, pandas, etc.).

`--no-cache-dir` reduce el tamaño de la imagen eliminando la cache de pip.

### 10.3 Docker Compose — nuestros servicios

| Servicio | Imagen | Funcion | Puerto |
|----------|--------|---------|--------|
| postgres | postgres:13 | Base de datos de metadata de Airflow | 5432 (interno) |
| airflow-webserver | Custom (Dockerfile) | Interfaz web | 8080 |
| airflow-scheduler | Custom (Dockerfile) | Ejecuta DAGs | - |
| airflow-triggerer | Custom (Dockerfile) | Deferrable operators | - |
| airflow-init | Custom (Dockerfile) | Inicializa DB + crea usuario admin | - |

### 10.4 Volumenes y por que son importantes

```yaml
volumes:
  - ./dags:/opt/airflow/dags          # DAG files
  - ../scripts:/opt/airflow/scripts   # Los 3 scripts
  - ../data:/opt/airflow/data         # dataset.csv + processed/
  - ../models:/opt/airflow/models     # churn_model.pkl
```

Los volumenes **montan** carpetas del host dentro del contenedor. Esto significa:
- Los cambios en los scripts se reflejan inmediatamente sin reconstruir la imagen.
- Los datos procesados y modelos se persisten en el host, sobreviviendo a `docker compose down`.
- El desarrollo es iterativo: editas → ejecutas → ves resultados.

### 10.5 Variables de entorno

```yaml
CHURN_DATA_DIR: /opt/airflow/data
CHURN_SCRIPTS_DIR: /opt/airflow/scripts
CHURN_MODELS_DIR: /opt/airflow/models
```

Las rutas son configurables via variables de entorno. Esto permite usar los mismos scripts tanto en Docker (con estas rutas) como localmente (con rutas relativas como `../data/`). El DAG lee estas variables con `os.environ.get()`.

---

## 11. ESTUDIO DE ABLACION — METODOLOGIA

### 11.1 Que es un estudio de ablacion

En ML, un estudio de ablacion consiste en eliminar componentes del modelo uno a uno para medir su contribucion individual. Se origina en neurociencia, donde se destruyen ("ablacionan") regiones del cerebro para entender su funcion.

### 11.2 Nuestro caso: change_mou

**Hipotesis**: change_mou podria ser leakage temporal si representa el cambio DESPUES de la decision de irse.

**Metodologia**:
1. Entrenar modelo A: con todas las features (incluida change_mou).
2. Entrenar modelo B: con todas las features EXCEPTO change_mou.
3. Evaluar ambos en el mismo test set.
4. Comparar AUC-ROC.

**Resultado**: AUC_A = 0.7023, AUC_B ≈ 0.67, Delta ≈ 3 pp.

**Regla de decision**: Si delta > 5-10 pp, la feature probablemente es leakage (el modelo depende demasiado de ella). Si delta < 5 pp, la feature es una señal legitima que mejora marginalmente.

**Conclusion**: 3 pp esta dentro del rango aceptable. change_mou captura comportamiento previo al churn (el cliente empieza a usar menos ANTES de cancelar) y es una señal legitima.

---

## 12. POSIBLES EXTENSIONES (BONUS)

### 12.1 MLflow

Plataforma de experiment tracking. Registra parametros, metricas, artefactos (modelos, graficos) de cada run. Permite comparar experimentos y versionar modelos con un Model Registry.

Se integraria en train_model.py: `mlflow.log_param("max_depth", 6)`, `mlflow.log_metric("auc", 0.70)`, `mlflow.sklearn.log_model(model, "churn_model")`.

### 12.2 PostgreSQL para datos

En vez de guardar los datos procesados como CSV, se podrian persistir en PostgreSQL (ya lo tenemos corriendo para Airflow). Ventajas: queries SQL directos, control de acceso, transacciones ACID.

### 12.3 Prometheus + Grafana

Prometheus recoge metricas de aplicaciones. Grafana las visualiza en dashboards. Para nuestro caso, monitorizariamos:
- AUC-ROC de cada reentrenamiento (detectar degradacion).
- Distribucion de features de entrada (detectar data drift).
- Tiempo de ejecucion del pipeline.
- Alertas si AUC cae por debajo de un umbral.

### 12.4 Feature Store

Repositorio centralizado de features reutilizables. En vez de que cada modelo tenga su propio script de feature engineering, las features se calculan una vez y se comparten. Herramientas: Feast, Tecton, AWS Feature Store.

---

## 13. PREGUNTAS FRECUENTES Y COMO RESPONDERLAS

### Sobre los datos

**P: "¿Por que eliminaste columnas en vez de imputarlas?"**
R: "Hay un umbral practico. Si una columna tiene mas del 85% de nulos, estaramos inventando el 85% de los datos. Cualquier patron que el modelo aprenda de esa columna seria artificial. Es mejor eliminarla y dejar que el modelo use variables con informacion real."

**P: "¿Consideraste tecnicas de oversampling como SMOTE?"**
R: "No fue necesario porque el dataset ya esta balanceado al 50/50. En produccion con datos reales (2% churn), si seria necesario. Las opciones serian SMOTE para oversampling, o simplemente ajustar scale_pos_weight en XGBoost, que es mas eficiente y no genera muestras sinteticas."

**P: "¿Y si hay datos nuevos que tienen categorias que no estaban en el train?"**
R: "LabelEncoder asignaria un error. En produccion, habria que manejar categorias desconocidas: o bien asignarlas a una categoria 'UNKNOWN', o usar un encoder que soporte categorias nuevas como OrdinalEncoder con handle_unknown='use_encoded_value'."

### Sobre el modelo

**P: "¿El AUC de 0.70 es bueno o malo?"**
R: "Depende del contexto. Para un dataset artificial con 100 features ruidosas y trampas, es un resultado solido. En la industria, modelos de churn con datos reales suelen estar entre 0.75-0.85. Con feature engineering adicional, datos temporales, y variables de contrato (que aqui no tenemos), podriamos mejorar."

**P: "¿Que harias para mejorar el AUC?"**
R: "Varias cosas: mas feature engineering (interacciones, lag features), datos temporales (tendencias de uso mes a mes), stacking de modelos, y sobre todo datos de mejor calidad (sin las trampas artificiales)."

**P: "¿Que pasa si el modelo se equivoca mucho?"**
R: "Los falsos positivos (contactar a alguien que no iba a irse) tienen un coste bajo: el cliente recibe una oferta que probablemente aprecia. Los falsos negativos (no detectar un churner) tienen un coste alto: perdemos un cliente. Por eso es preferible un modelo con alto recall aunque baje la precision, y ajustar el threshold segun el coste relativo."

### Sobre la arquitectura

**P: "¿Por que Airflow y no un simple cron job?"**
R: "Un cron job ejecuta scripts pero no ofrece: visualizacion del grafo de dependencias, reintentos automaticos, logging centralizado, interfaz web para monitorizar, alertas por fallo, ni historial de ejecuciones. Airflow da todo esto out-of-the-box."

**P: "¿Como desplegarias esto en la nube?"**
R: "En AWS: Amazon MWAA (Managed Airflow) + S3 para datos + RDS para PostgreSQL + ECR para la imagen Docker. En GCP: Cloud Composer + GCS + CloudSQL. Ambos son Airflow gestionado que elimina la operacion del cluster."

**P: "¿Que pasaria si el dataset fuera de 100 millones en vez de 100 mil?"**
R: "Habria que cambiar varias cosas: usar Spark o Dask para el preprocesamiento en vez de pandas, considerar entrenamiento distribuido con Dask-XGBoost, y posiblemente CeleryExecutor o KubernetesExecutor en Airflow para paralelizar."

---

## 14. CHECKLIST FINAL ANTES DE LA PRESENTACION

- [ ] Ejecutar Docker localmente y verificar que el pipeline funciona end-to-end.
- [ ] Tener los notebooks (01_EDA, 02_EDA_Profundo, 03_Modeling) abiertos por si piden ver codigo.
- [ ] Repasar este documento, especialmente las secciones de metricas y SHAP.
- [ ] Preparar respuestas a las preguntas frecuentes de la seccion 13.
- [ ] Ensayar la presentacion cronometrada (objetivo: 30 minutos sin prisas).
- [ ] Tener abierta la UI de Airflow (localhost:8080) para hacer una demo en vivo si lo piden.
- [ ] Revisar que los numeros de la presentacion coinciden con los del notebook (AUC, F1, etc.).
