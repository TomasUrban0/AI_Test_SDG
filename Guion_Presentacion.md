# Guión de Presentación — Churn Prediction SDG Group

**Duración estimada: 30-35 minutos**
**28 diapositivas | ~1.5 min por diapositiva de contenido, ~30s por diapositiva de sección**

---

## Diapositiva 1 — Portada (30 segundos)

**Lo que se ve:** Titulo "Predicción de Churn en Telecomunicaciones", tu nombre, fecha.

**Que decir:**

> "Buenos dias/tardes. Soy Miguel Ángel Montero y voy a presentar mi solución a la prueba técnica de predicción de churn en telecomunicaciones. Voy a recorrer todo el ciclo de vida del proyecto: desde la exploración inicial de los datos, pasando por la detección de peculiaridades en el dataset, la construcción y evaluación del modelo predictivo, hasta la arquitectura de produccion con Airflow y Docker."

---

## Diapositiva 2 — Agenda (45 segundos)

**Lo que se ve:** 6 bloques numerados con los temas.

**Que decir:**

> "He estructurado la presentación en seis bloques. Primero vamos a entender el problema de negocio y por que es importante. Después entraremos en el análisis exploratorio, donde veremos que el dataset tenia varias peculiaridades intencionadas que habia que detectar. Luego pasamos al modelo predictivo: por que elegimos XGBoost, como lo tuneamos y que resultados obtuvimos. Después traduciremos esos resultados a valor de negocio con la curva de lift y recomendaciones accionables. Veremos la arquitectura del pipeline con Airflow y Docker. Y finalmente, conclusiónes y siguientes pasos."

---

## Diapositiva 3 — Sección: Contexto del Problema (15 segundos)

**Lo que se ve:** Titulo de seccion sobre fondo oscuro.

**Que decir:**

> "Empecemos por entender el problema que estamos resolviendo."

---

## Diapositiva 4 — El Problema: Churn en Telecomunicaciones (2 minutos)

**Lo que se ve:** Texto explicativo a la izquierda, 3 tarjetas con métricas clave a la derecha (100K, 100, 50/50).

**Que decir:**

> "El churn, o la tasa de abandono de clientes, es uno de los problemas más costosos en telecomunicaciones. Se estima que captar un nuevo cliente cuestá entre 5 y 7 veces más que retener uno existente. En telco, la tasa de churn mensual típica está entre el 1 y el 3%, y cada punto porcentual de reduccion puede representar millones de euros en ingresos retenidos.
>
> Nuestro objetivo es construir un modelo que identifique a los clientes con mayor probabilidad de irse, ANTES de que se vayan, para poder actuar con campañas de retención dirigidas.
>
> El dataset que nos proporcionaron tiene 100.000 clientes, 100 variables que incluyen datos de uso, facturación, llamadas a soporte, antiguedad del equipo, etc. Un detalle importante: el dataset está balanceado artificialmente al 50/50 entre churn y no-churn. En la realidad, el churn suele ser del 1-3%, así que este balance es artificial. Esto lo tendremos en cuenta a la hora de interpretar métricas como accuracy."

**Posibles preguntas:**
- *"Por que importa que este balanceado artificialmente?"* → Porque accuracy será engañosamente alta. En produccion con datos reales (2% churn) necesitaríamos ajustar con class_weight o SMOTE.

---

## Diapositiva 5 — Seccion: Análisis Exploratorio (15 segundos)

> "Pasemos al análisis exploratorio, que fue la parte más reveladora del proyecto."

---

## Diapositiva 6 — Radiografía del Dataset (1.5 minutos)

**Lo que se ve:** 3 tarjetas de estadísticas principales (100K filas x 100 cols, 43 columnas con nulos, 50.4/49.6% churn/no-churn), panel izquierdo con composicion de variables por tipo, panel derecho con top correlaciónes con churn.

**Que decir:**

> "Antes de entrar en los hallazgos, veamos la radiografia general del dataset.
>
> Tenemos **100.000 filas y 100 columnas**. De esas 100 columnas, 69 son float, 10 son enteras y 21 categóricas. Esto ya nos dice que la mayoría de la información es numerica continua: facturación, minutos de uso, etc.
>
> **43 columnas contienen nulos**, pero ojo: muchos de estos nulos estan camuflados con valores como 'U' o 'Z', que parecen categorías validas pero en realidad son datos faltantes. Esto es una trampa clásica que un pipeline automático no detectaria con isnull(). No hay filas duplicadas, lo cual es bueno.
>
> El balance de churn es **50.4% vs 49.6%**, prácticamente 50/50. Como ya comente, esto es artificial — en la realidad el churn está en el 1-3%. El balance nos facilita el entrenamiento pero hay que recordarlo al interpretar resultados.
>
> En cuanto a **correlaciónes con churn**, las más fuertes son sorprendentemente bajas: eqpdays (dias de antiguedad del equipo) con +0.113, hnd_price (precio del terminal) con -0.103, y totmrc_Mean (facturación recurrente media) con -0.069. Que la correlación maxima sea de solo 0.11 nos dice que ningun feature individual predice churn por si solo — el valor está en la combinacion de señales, que es donde XGBoost brilla.
>
> El perfil del churner que emerge es: equipo antiguo, terminal barato, menor gasto mensual, menos minutos de uso y menos llamadas completadas. Es un cliente desenganchado."

**Posibles preguntas:**
- *"¿Por que las correlaciónes son tan bajas?"* → Porque churn es un fenomeno multifactorial. Ningun factor aislado lo explica, por eso necesitamos modelos que capturen interacciones no lineales entre variables.

---

## Diapositiva 7 — EDA: Hallazgos Principales (2.5 minutos)

**Lo que se ve:** 4 tarjetas con hallazgos clave: balance artificial, nulos camuflados, multicolinealidad, valores negativos.

**Que decir:**

> "El EDA revelo que el dataset contenia varias peculiaridades intencionadas. Vamos una por una.
>
> **Balance artificial 50/50**: como ya mencione, tener exactamente 50% churn es poco realista. Esto nos dice que el dataset fue construido para propósitos de testing. La ventaja es que no necesitamos técnicas de oversampling como SMOTE, pero debemos recordar que en produccion las proporciones serán muy diferentes.
>
> **Nulos camuflados**: las columnas kid0_2 hasta kid16_17 usaban la letra 'U' como valor en más del 90% de los casos. Otros campos usaban 'Z'. Estos no son valores reales, son nulos disfrazados que un pipeline automático no detectaria con isnull(). Los reemplazamos por NaN antes de cualquier análisis.
>
> **Multicolinealidad extrema**: encontramos pares de variables con correlación de 1.000, como rev_Mean y totrev, que son literalmente la misma información. Tener ambas es redundante y puede causar inestabilidad en modelos lineales. Eliminamos la del par que tenia menor correlación con churn, usando un umbral de r > 0.98.
>
> **Valores negativos en eqpdays**: eqpdays representa los dias de antiguedad del equipo del cliente. Encontramos valores negativos, lo cual es físicamente imposible. Los convertimos a NaN y se imputaron con la mediana."

**Posibles preguntas:**
- *"Como detectaste los nulos camuflados?"* → Analizando las distribuciones de frecuencia de las categóricas. Cuando una categoría tiene >90% del mismo valor, es sospechoso.
- *"Por que mediana y no media para imputar?"* → La mediana es robusta a outliers. En variables sesgadas (como ingresos), la media estaria distorsionada.

---

## Diapositiva 8 — Peculiaridades Detectadas en el Dataset (2 minutos)

**Lo que se ve:** 5 filas con badges de severidad (CRITICA, MEDIA, VALIDADA, ALTA).

**Que decir:**

> "Además de los hallazgos del EDA basico, detectamos peculiaridades adicionales.
>
> La más **critica**: Customer_ID está ordenado y correlaciónado con churn. Si no lo eliminamos, el modelo aprenderia el orden de los IDs en vez de patrones reales. Esto seria data leakage puro.
>
> **Varianza casi nula**: variables donde más del 99% de las filas tienen el mismo valor. No aportan capacidad discriminativa y solo añaden ruido.
>
> **Dummies sin señal**: variables binarias donde la tasa de churn es idéntica para 0 y para 1. Es decir, no tienen correlación alguna con el target.
>
> Un caso interesante: **change_mou**, que es el cambio en minutos de uso. Es el predictor número 1 segun SHAP, pero nos preguntamos si podría ser leakage: si representa el cambio DESPUÉS de que el cliente decide irse, seria data leakage. Hicimos un estudio de ablacion que veremos más adelante, y confirmamos que NO es leakage.
>
> Y finalmente, eliminamos todas las columnas con más del 85% de nulos. Imputar más del 85% seria basicamente inventar datos."

---

## Diapositiva 9 — Feature Engineering: 8 Variables Nuevas (2 minutos)

**Lo que se ve:** Tabla con las 8 features, su descripción y lógica de calculo.

**Que decir:**

> "Tras la limpieza, creamos 8 features nuevas, todas con interpretación de negocio clara.
>
> **rev_per_minute**: ingreso por minuto de uso. Mide cuanto paga el cliente por cada minuto que habla. Un valor alto puede indicar un plan caro relativo al uso.
>
> **call_fail_rate**: porcentaje de llamadas que fallan (caidas + bloqueadas sobre intentos totales). Si un cliente sufre muchos fallos de red, su experiencia se degrada y es más probable que se vaya.
>
> **care_intensity**: ratio de llamadas al servicio de atencion al cliente sobre el uso total. Un cliente que llama mucho a soporte pero usa poco el servicio probablemente está frustrado.
>
> **old_device**: flag binaria, 1 si el equipo tiene más de un año. Los clientes con equipos viejos suelen estar menos comprometidos o esperando a que expire el contrato.
>
> **usage_drop_severe**: 1 si el cambio en minutos de uso es menor a -50. Una caida severa es una señal de desenganche.
>
> **rev_vs_avg3**: ratio entre la facturación actual y la media de los ultimos 3 meses. Detecta cambios bruscos en el patron de gasto.
>
> **new_customer**: 1 si el cliente tiene menos de 6 meses. Los clientes nuevos son más volátiles.
>
> **engagement_score**: score compuesto que combina el ranking de uso, llamadas completadas y recibidas. Da una vision holistica del engagement."

**Posibles preguntas:**
- *"Por que no usaste PCA o selección automatica de features?"* → Porque el objetivo era mantener la interpretabilidad. SHAP sobre features interpretables da insights accionables para el negocio.

---

## Diapositiva 10 — Sección: Modelo Predictivo (15 segundos)

> "Pasemos a la parte de modelado."

---

## Diapositiva 11 — Pipeline de Preprocesamiento (1.5 minutos)

**Lo que se ve:** 5 tarjetas en flujo horizontal: Limpieza → FE → LabelEncoder → Imputación → Split.

**Que decir:**

> "Antes de entrenar, el preprocesamiento sigue estos 5 pasos en orden.
>
> Primero, la limpieza de todas las peculiaridades que vimos. Después, el feature engineering con las 8 variables nuevas. Luego, LabelEncoder para convertir las categóricas a numerico — elegimos LabelEncoder sobre OneHotEncoder porque con árboles de decisión (XGBoost) funciona igual de bien y evita la explosión de dimensiónalidad.
>
> La imputación de nulos numericos se hace con la mediana, que es robusta a outliers. Y finalmente, un split estratificado 70/15/15: train, validation y test. El validation set se usa durante el entrenamiento para early stopping y sanity check, y el test set se reserva exclusivamente para la evaluación final.
>
> El resultado son 100.000 filas por 88 columnas limpias, sin leakage."

**Posibles preguntas:**
- *"Por que 70/15/15 y no 80/20?"* → Necesitamos un validation set separado para tuning y monitoring durante el entrenamiento, además del test set intocable.
- *"Por que no StandardScaler?"* → XGBoost es invariante a escala porque usa árboles. El scaler se guarda en los artefactos por si usamos Logistic Regression.

---

## Diapositiva 12 — Benchmarking: 5 Modelos Evaluados (2 minutos)

**Lo que se ve:** Gráfico de barras horizontales con AUC-ROC de 5 modelos + panel explicativo "Por que XGBoost".

**Que decir:**

> "Probamos 5 modelos en orden de complejidad creciente.
>
> **Logistic Regression** como baseline: AUC de 0.626. Es el piso mínimo que cualquier modelo debería superar. Lo mantenemos como referencia interpretable.
>
> **Decision Tree**: 0.638. Mejora ligeramente pero sufre de overfitting.
>
> **Random Forest**: 0.671. El ensemble ya mejora significativamente al arbol individual.
>
> **XGBoost y LightGBM empatan en 0.694**. Ambos son gradient boosting, pero elegimos XGBoost por varias razones: tiene regularización L1/L2 nativa que ayuda a controlar overfitting, tiene mejor soporte para SHAP con TreeExplainer optimizado, tiene un ecosistema más maduro y documentado, y el parametro scale_pos_weight nos da control directo sobre el desbalance de clases cuando lo necesitemos en produccion.
>
> La Logistic Regression se mantiene como baseline interpretable. La diferencia de casi 7 puntos de AUC entre LR y XGBoost justifica la complejidad adicional."

**Posibles preguntas:**
- *"Probaste redes neuronales?"* → No, porque con datos tabulares de 100K filas, los gradient boosting trees son consistentemente superiores segun la literatura. Además, perderíamos interpretabilidad con SHAP.

---

## Diapositiva 13 — Hyperparameter Tuning (1.5 minutos)

**Lo que se ve:** Panel izquierdo con parámetros de configuración, panel derecho con razonamiento.

**Que decir:**

> "Para el tuning usamos RandomizedSearchCV con 30 iteraciones y 3-fold cross-validation.
>
> ¿Por que RandomizedSearch y no GridSearch? Con 11 hiperparámetros, el grid completo tendría miles de combinaciones. Bergstra y Bengio demostraron en 2012 que el random search es más eficiente porque explora el espacio de forma más uniforme, especialmente cuando no todos los parámetros son igualmente importantes.
>
> Usamos 3-fold porque con 100.000 filas, cada fold tiene unas 67.000 muestras, que es más que suficiente para una estimación estable. 5-fold triplicaría el tiempo sin mejorar significativamente la estimación.
>
> La métrica objetivo es AUC-ROC porque es threshold-independent. Optimizar AUC nos da el mejor modelo posible sin comprometernos a un punto de corte fijo, que luego ajustaremos segun el coste de negocio.
>
> Los parámetros de regularización (reg_alpha y reg_lambda) son clave: con 88 features, queremos controlar el overfitting."

---

## Diapositiva 14 — Métricas Finales en Test Set (2.5 minutos)

**Lo que se ve:** 4 tarjetas grandes con AUC-ROC 0.6986, F1 0.6399, Recall 0.6415, Precision 0.6382 + confusion matrix + interpretación.

**Que decir:**

> "Estos son los resultados finales sobre el test set, que el modelo nunca vio durante el entrenamiento.
>
> **AUC-ROC de 0.6986**: esto indica una capacidad discriminativa moderada-buena. Puede parecer modesto, pero hay que tener en cuenta que estamos trabajando con un dataset de 100 columnas llenas de peculiaridades y ruido. Un AUC de ~0.70 sobre datos limpios y con features interpretables es un resultado solido.
>
> **F1 Score de 0.6399**: equilibra precision y recall. En nuestro caso, ambas estan bastante parejas, lo cual es bueno.
>
> **Recall de 0.6415**: de cada 100 clientes que realmente van a hacer churn, nuestro modelo identifica ~64. Los otros 36 se nos escapan. En negocio, esto significa que capturamos casi dos tercios de los churners.
>
> **Precision de 0.6382**: de cada 100 clientes que el modelo marca como "va a hacer churn", ~64 realmente lo hacen. Los otros 36 son falsos positivos, clientes que recibirian una oferta de retención innecesaria.
>
> Si miramos la confusion matrix: de los ~7.434 clientes en test que realmente hicieron churn, identificamos 4.769 (los true positives). Y de los 7.566 que no hicieron churn, clasificamos correctamente 4.863.
>
> Un punto importante: el threshold de 0.50 es ajustable. Dependiendo de si el negocio prefiere maximizar recall (no perder ningun churner, aunque contactemos a algunos que no lo son) o precision (contactar solo a los seguros), podemos moverlo."

---

## Diapositiva 15 — SHAP: Interpretabilidad del Modelo (2 minutos)

**Lo que se ve:** Ranking de top 10 features por importancia SHAP (barras) + panel de interpretación.

**Que decir:**

> "SHAP, o SHapley Additive exPlanations, nos permite entender por que el modelo toma cada decisión. Está basado en la teoria de juegos cooperativos de Shapley: cada feature recibe un valor que representa su contribución marginal a la prediccion.
>
> El predictor más fuerte es **change_mou**, el cambio en minutos de uso. Tiene todo el sentido: un cliente que reduce drásticamente su uso probablemente está considerando cambiar de proveedor.
>
> **mou_Mean** (minutos de uso promedio) y **eqpdays** (antiguedad del equipo) son el segundo y tercer predictor. Clientes con poco uso y equipos viejos tienen mayor riesgo.
>
> **months** (antiguedad del cliente) aparece cuarto: clientes nuevos, con menos de 6 meses, son significativamente más volátiles.
>
> Un dato interesante: **call_fail_rate**, una de nuestras features nuevas de feature engineering, aparece en el top 8. Esto valida que la calidad de servicio percibida impacta directamente en churn.
>
> Lo más poderoso de SHAP es que no solo da importancias globales: podemos generar explicaciones por cliente individual. 'Este cliente tiene riesgo 78% porque su uso cayó un 60%, tiene el equipo más viejo, y ha llamado 5 veces al servicio técnico.' Eso es lo que el equipo de retención necesita para personalizar la intervención."

**Posibles preguntas:**
- *"Cual es la diferencia entre SHAP y feature importance de XGBoost?"* → La feature importance nativa de XGBoost está basada en gain y puede ser inconsistente. SHAP satisface propiedades axiomaticas (eficiencia, simetria, dummy) que la hacen teoricamente más robusta.

---

## Diapositiva 16 — Sección: Valor de Negocio (15 segundos)

> "Ahora traduzcamos estos números a impacto de negocio."

---

## Diapositiva 17 — Curva de Lift: Impacto en Negocio (2 minutos)

**Lo que se ve:** Gráfico de línea con curva de lift vs aleatorio + panel de impacto practico.

**Que decir:**

> "La curva de lift es probablemente la métrica más relevante para el negocio. Responde a una pregunta directa: si solo puedo contactar a un porcentaje de mis clientes, ¿cuantos churners voy a capturar?
>
> La línea azul es nuestro modelo y la gris es el baseline aleatorio. Si contactamos al azar al 10% de los clientes, capturaremos el 10% de los churners. Con nuestro modelo, contactando al top 10% de clientes por riesgo segun el modelo, capturamos un ~16% de los churners. Eso es un **lift de 1.59x**, es decir, un 59% más eficiente que contactar al azar.
>
> En la práctica, si una telco tiene 10 millones de clientes y un churn mensual del 2%, eso son 200.000 churners al mes. Contactar al azar al 10% (1 millon de clientes) capturaria 20.000. Con nuestro modelo, capturaríamos ~31.800 con el mismo esfuerzo. Son ~11.800 clientes más retenidos por mes.
>
> Si el ARPU (ingreso medio por usuario) es de 30 euros al mes, esos 12.000 clientes extra representan 360.000 euros mensuales en ingresos retenidos. Descontando el coste de las campañas de retención, el ROI suele ser positivo contactando al top 20-30%."

---

## Diapositiva 18 — Estudio de Ablacion: change_mou (1.5 minutos)

**Lo que se ve:** Explicacion del problema, tabla comparativa con/sin change_mou, conclusión con check verde.

**Que decir:**

> "Este es un punto que quiero destacar porque demuestra rigor analitico. change_mou es el predictor número 1, pero nos planteamos una pregunta crítica: ¿podría ser leakage?
>
> El argumento seria: si change_mou representa el cambio en uso DESPUÉS de que el cliente ya decidio irse, entonces estaríamos incurriendo en data leakage. El modelo estaria usando información del futuro para predecir algo que ya pasó.
>
> Para verificarlo, hicimos un estudio de ablacion: entrenamos el modelo completo quitando change_mou y comparamos las métricas. El AUC cayó de 0.6986 a aproximadamente 0.67, un delta de solo ~3 puntos porcentuales. El F1 cayó similarmente unos 2 puntos.
>
> Nuestra regla: si el delta fuera mayor a 5 puntos porcentuales, seria sospechoso de leakage. Con un delta de 3pp, concluimos que change_mou es una señal legitima de comportamiento previo al churn, no información del futuro. El cliente empieza a usar menos el servicio antes de irse, y eso es lo que estamos capturando."

---

## Diapositiva 19 — Recomendaciones de Negocio (2 minutos)

**Lo que se ve:** 5 recomendaciones con iconos.

**Que decir:**

> "Basándonos en los insights del modelo y SHAP, tenemos 5 recomendaciones accionables.
>
> **Uno: campañas de retención focalizadas**. Usar el modelo para priorizar al top 20-30% de clientes por riesgo. Con el lift de 1.5x, cada euro invertido en retención rinde un 50% más que contactar al azar.
>
> **Dos: programa de renovacion de equipos**. eqpdays es un predictor fuerte. Los clientes con equipos viejos (mas de un año) tienen mayor probabilidad de churn. Un programa de upgrade subsidiado podría retenerlos.
>
> **Tres: alertas de calidad de red**. call_fail_rate es un predictor significativo. Si monitorizamos la calidad de red por zona y actuamos proactivamente ante degradaciónes, podemos prevenir churn antes de que el cliente llame a quejarse.
>
> **Cuatro: onboarding reforzado**. Los clientes nuevos (menos de 6 meses) son los más volátiles. Un programa de bienvenida con incentivos en los primeros 90 dias puede marcar la diferencia.
>
> **Cinco: scoring semanal automatizado**. Ejecutar el pipeline periódicamente, alimentar el CRM con scores de riesgo actualizados, y generar listas de accion para el equipo de retención."

---

## Diapositiva 20 — Sección: Arquitectura del Pipeline (15 segundos)

> "Veamos como se lleva todo esto a un entorno de produccion."

---

## Diapositiva 21 — Estructura del Proyecto y Modularización (1.5 minutos)

**Lo que se ve:** Panel izquierdo con el arbol de carpetas del proyecto en fuente monoespaciada (scripts/, dags/, data/, models/, monitoring/, docker-compose.yaml, Dockerfile, requirements.txt). Panel derecho con 4 tarjetas categorizadas: CORE (azul), UTILS (verde), INFRA (morado), DESIGN (ambar).

**Que decir:**

> "Antes de ver el DAG y Docker en detalle, quiero mostrar como está organizado el código, porque la modularización fue una decisión de diseño deliberada.
>
> A la izquierda teneis el arbol del proyecto. La carpeta **scripts/** contiene 5 archivos Python. Los 3 scripts core — data_preparation, train_model y evaluate_model — siguen un patron comun: cada uno tiene una función main() independiente que encapsula toda su lógica. El DAG simplemente importa y ejecuta cada main() via PythonOperator. Esto significa que los scripts se pueden ejecutar tanto dentro de Airflow como de forma aislada desde la terminal, lo cual facilita mucho el debugging.
>
> Los **modulos reutilizables** — db_utils.py y metrics_exporter.py — encapsulan la lógica de conexión a PostgreSQL y el push de métricas a Prometheus respectivamente. Estan desacoplados de los scripts core: si PostgreSQL no está disponible, el pipeline sigue funcionando porque cada extensión tiene un fallback graceful.
>
> La **infraestructura está completamente codificada**: docker-compose.yaml define los 10 servicios, el Dockerfile hereda de la imagen oficial de Airflow y añade las dependencias ML, y los volumenes montan scripts/ y data/ directamente en el contenedor para hot-reload sin rebuild.
>
> Un detalle importante: el **pickle es auto-contenido**. Incluye no solo el modelo XGBoost sino también los encoders, el imputer y el scaler. Esto permite que evaluate_model.py pueda hacer predicciones sin necesidad de reejecutar el preprocesamiento."

**Posibles preguntas:**
- *"Por que no usaste un único script monolitico?"* → Porque la separacion en 3 scripts permite reejecutar pasos individuales (ej: re-evaluar sin re-entrenar), facilita el testing, y cada task de Airflow tiene visibilidad independiente en los logs.
- *"Por que fallback en vez de lanzar error?"* → Porque las extensiónes (DB, MLflow, Prometheus) son nice-to-have, no bloqueantes. El pipeline core debe funcionar siempre. Es un patron comun en produccion: degradación graceful.

---

## Diapositiva 22 — Apache Airflow: DAG de 3 Tasks (2 minutos)

**Lo que se ve:** 3 tarjetas en flujo (data_preparation → train_model → evaluate_model) con tiempos de ejecución.

**Que decir:**

> "El pipeline está orquestado con Apache Airflow en un DAG de exactamente 3 tasks, como pide la prueba.
>
> **Task 1: data_preparation**, que tarda unos 16 segundos. Carga el dataset original, ejecuta toda la limpieza de peculiaridades, el feature engineering, el encoding y el split. Guarda los datasets procesados como CSV y los artefactos de preprocesamiento como pickle.
>
> **Task 2: train_model**, unos 5 segundos. Carga los datos procesados, entrena el XGBoost con los hiperparámetros optimizados, hace un sanity check en el validation set, y guarda el modelo completo como pickle. El pickle incluye no solo el modelo sino también los artefactos de preprocesamiento (encoders, imputer, scaler) para que sea auto-contenido.
>
> **Task 3: evaluate_model**, menos de 1 segundo. Carga el pickle y el test set, genera predicciones, calcula todas las métricas (AUC-ROC, F1, confusion matrix, lift curve), las imprime en los logs de Airflow, y guarda un JSON con el reporte completo. Además tiene un quality gate: si el AUC cae por debajo de 0.55, el task falla.
>
> Usamos PythonOperator porque cada task importa y ejecuta la función main() del script correspondiente. Si un paso falla, el DAG se detiene y no ejecuta los siguientes."

---

## Diapositiva 23 — Infraestructura: Docker Compose (1.5 minutos)

**Lo que se ve:** 10 servicios de Docker, panel de decisiones de diseño, comandos.

**Que decir:**

> "Todo corre sobre Docker Compose con 10 servicios: PostgreSQL como base de datos de metadata de Airflow y también como store de churn_db (metricas, predicciones, features) y mlflow_db, el servidor MLflow para experiment tracking en el puerto 5000, el Webserver para la interfaz grafica en el puerto 8080, el Scheduler que ejecuta los DAGs, el Triggerer para operadores diferibles, airflow-init para la inicializacion, y el stack completo de monitorización con **Pushgateway** en el puerto 9091, **Prometheus** en el puerto 9090 y **Grafana** en el puerto 3000.
>
> El stack de monitorización funciona asi: al final de cada evaluate_model, el modulo metrics_exporter.py envia las métricas del run al Pushgateway. Prometheus scrapea el Pushgateway cada 15 segundos. Y Grafana visualiza todo en un dashboard llamado 'Churn Prediction Pipeline' con 6 paneles: timeseries de AUC-ROC y F1, estadísticas actuales, confusion matrix, lift por percentil, y datos del pipeline.
>
> Decisiones de diseño importantes: elegimos **LocalExecutor** en vez de CeleryExecutor porque para un solo DAG secuencial no necesitamos la complejidad de Celery + Redis. Esto reduce el consumo de RAM de 8GB a unos 4GB.
>
> La imagen es **custom**: la imagen base de Airflow no incluye xgboost, scikit-learn ni las otras dependencias de ML. Nuestro Dockerfile hereda de la imagen oficial y ejecuta pip install de nuestro requirements.txt.
>
> Los **volumenes** montan las carpetas scripts/, data/ y models/ directamente en el contenedor. Esto significa que los cambios en los scripts se reflejan inmediatamente sin reconstruir la imagen.
>
> Para ejecutar todo, son literalmente 3 comandos: build, init, up. Y las interfaces quedan disponibles en localhost:8080 (Airflow), localhost:5000 (MLflow), y localhost:3000 (Grafana)."

---

## Diapositiva 24 — Evidencia: Orquestación y Experiment Tracking (1.5 minutos)

**Lo que se ve:** Capturas de pantalla reales de Airflow (DAG con 10 runs verdes) y MLflow (run completado con métricas).

**Que decir:**

> "Esto es evidencia real de la infraestructura funcionando. A la izquierda veis Apache Airflow con el DAG churn_prediction_pipeline: 10 ejecuciónes exitosas, cada una con 3 tasks verdes. El ultimo run tardo 1 minuto y 21 segundos de principio a fin.
>
> A la derecha, MLflow 2.12.2 mostrando el detalle de un run completado. Podeis ver los 41 hiperparámetros del modelo registrados automáticamente, y las 4 métricas de validación: val_auc_roc de 0.6913, val_f1 de 0.6376, val_recall de 0.6434 y val_precision de 0.6319. Todo queda versiónado para comparación entre runs."

---

## Diapositiva 25 — Evidencia: Monitoreo con Prometheus y Grafana (1.5 minutos)

**Lo que se ve:** Captura grande del dashboard de Grafana con métricas reales + captura de Prometheus con la query churn_test_auc_roc.

**Que decir:**

> "Y está es la pieza final del stack de monitorización. El dashboard de Grafana 'Churn Prediction Pipeline' tiene 6 paneles: arriba las timeseries de AUC-ROC y F1 Score, debajo las métricas actuales del modelo — 69.9% AUC-ROC, 64% F1 — la confusion matrix con los 4.77K true positives y 4.86K true negatives, el lift por percentil mostrando 1.67x en el top 5%, y los datos del pipeline como las 15K muestras de test.
>
> A la derecha, Prometheus confirmando que tiene los datos: la query churn_test_auc_roc devuelve el valor 0.6986 para los dos runs del job churn_pipeline.
>
> El flujo completo es: evaluate_model ejecuta metrics_exporter.py que hace un HTTP PUT al Pushgateway en el puerto 9091. Prometheus scrapea el Pushgateway cada 15 segundos. Y Grafana consulta a Prometheus para renderizar los paneles. Todo automático, sin intervención manual."

---

## Diapositiva 26 — Sección: Conclusiónes (15 segundos)

> "Para cerrar, un resumen de lo construido y los siguientes pasos."

---

## Diapositiva 27 — Resumen y Siguientes Pasos (2 minutos)

**Lo que se ve:** Checklist de lo construido (izquierda) + lista de siguientes pasos (derecha).

**Que decir:**

> "Resumiendo lo que hemos construido: un EDA profundo con detección de más de 13 peculiaridades, 8 features nuevas con lógica de negocio, un XGBoost tuneado con AUC de ~0.70 y lift de 1.59x, interpretabilidad con SHAP, un estudio de ablacion para validar que no hay leakage, 3 scripts modulares y testeados end-to-end, y un DAG de Airflow con Docker listo para desplegar.
>
> Pero además del pipeline base, he implementado tres extensiónes que demuestran capacidad MLOps:
>
> **PostgreSQL** como capa de persistencia: las métricas, predicciones y features procesadas se guardan automáticamente en 3 tablas de la base de datos churn_db. Esto permite consultas SQL directas, historial de runs para detectar degradación, y alimentar un CRM con scores de riesgo de churn. El run_id se comparte entre las 3 tasks del DAG via XCom para garantizar consistencia.
>
> **MLflow** como plataforma de experiment tracking: cada ejecución del pipeline registra automáticamente los hiperparámetros del modelo, las métricas de validación y test, la confusion matrix, la curva de lift, y el modelo serializado. La UI de MLflow permite comparar runs y detectar regresiones.
>
> **Prometheus + Grafana** como stack de monitorización: al final de cada evaluate_model, las métricas se envian automáticamente al Pushgateway via metrics_exporter.py. Prometheus las scrapea cada 15 segundos y Grafana las visualiza en un dashboard 'Churn Prediction Pipeline' con 6 paneles: timeseries de AUC-ROC y F1, estadísticas actuales, confusion matrix, lift por percentil y datos del pipeline. Accesible en localhost:3000.
>
> Para **siguientes pasos**, las extensiónes que aportarian más valor serian:
>
> **Feature store** para centralizar features reutilizables entre equipos y modelos.
>
> **Alertas automáticas en Grafana** para notificar via Slack o email cuando el AUC-ROC cae por debajo de un umbral configurable.
>
> Y cuando trabajemos con **datos reales desbalanceados** (churn del 2%), tendremos que adaptar con class_weight o SMOTE y recalibrar los thresholds."

---

## Diapositiva 28 — Gracias / Q&A (abierto)

**Lo que se ve:** "Gracias", "Preguntas y Respuestas", contacto.

**Que decir:**

> "Con esto concluyo la presentación. Quedo abierto a cualquier pregunta sobre el EDA, el modelo, la arquitectura, o cualquier decisión técnica que quieran profundizar. Gracias."

---

## Apendice: Preguntas Frecuentes Anticipadas

### Sobre los datos
- **"¿Que harías diferente con datos reales?"** → Primero, SMOTE o class_weight para el desbalance. Segundo, validación temporal (train en meses anteriores, test en el ultimo mes). Tercero, analisis de drift periodico.
- **"¿Por que no eliminaste más variables?"** → Mantuvimos todas las que pasaron los filtros de varianza, multicolinealidad y señal. XGBoost maneja bien features ruidosas gracias a su regularización.

### Sobre el modelo
- **"¿Por que no deep learning?"** → Para datos tabulares de este tamaño, los gradient boosting trees son state-of-the-art segun benchmarks recientes (Grinsztajn et al., 2022). Además, SHAP TreeExplainer es exacto, mientras que SHAP en redes neuronales es aproximado.
- **"¿El AUC de 0.70 es suficiente?"** → Depende del contexto. Para un dataset con 100 features ruidosas y peculiaridades, es solido. En produccion, con feature engineering adicional, datos temporales, y variables de contrato, podría mejorar a 0.75-0.80.
- **"¿Que pasa si el modelo se degrada con el tiempo?"** → Model drift. Ya tenemos la infraestructura completa para detectarlo: cada run del pipeline guarda las métricas en PostgreSQL (tabla model_runs) y en MLflow, y además las envia a Prometheus via Pushgateway. El dashboard de Grafana "Churn Prediction Pipeline" muestra la evolucion temporal del AUC-ROC y F1, permitiendo detectar degradación visualmente. El siguiente paso seria configurar alertas automáticas en Grafana para notificar cuando el AUC caiga por debajo de un umbral.

### Sobre la arquitectura
- **"¿Por que Airflow y no Luigi/Prefect/Dagster?"** → Airflow es el estándar de la industria, tiene la mayor comunidad, integración nativa con la mayoría de clouds, y es lo que pide la prueba técnica.
- **"¿Como escalarias esto?"** → CeleryExecutor con Redis para paralelismo, Kubernetes para auto-scaling, y separar el entrenamiento en una máquina con GPU si es necesario.
- **"¿Por que PythonOperator y no BashOperator?"** → PythonOperator nos da mejor control de errores, logging integrado, y paso de contexto entre tasks via XCom si lo necesitamos.
