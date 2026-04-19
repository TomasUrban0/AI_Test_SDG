# Guion de Presentacion — Churn Prediction SDG Group

**Duracion estimada: 30-35 minutos**
**25 diapositivas | ~1.5 min por diapositiva de contenido, ~30s por diapositiva de seccion**

---

## Diapositiva 1 — Portada (30 segundos)

**Lo que se ve:** Titulo "Prediccion de Churn en Telecomunicaciones", tu nombre, fecha.

**Que decir:**

> "Buenos dias/tardes. Soy Tomas Montero y voy a presentar mi solucion a la prueba tecnica de prediccion de churn en telecomunicaciones. A lo largo de esta presentacion voy a recorrer todo el ciclo de vida del proyecto: desde la exploracion inicial de los datos, pasando por la deteccion de trampas en el dataset, la construccion y evaluacion del modelo predictivo, hasta la arquitectura de produccion con Airflow y Docker."

---

## Diapositiva 2 — Agenda (45 segundos)

**Lo que se ve:** 6 bloques numerados con los temas.

**Que decir:**

> "He estructurado la presentacion en seis bloques. Primero vamos a entender el problema de negocio y por que es importante. Despues entraremos en el analisis exploratorio, donde veremos que el dataset tenia varias trampas intencionadas que habia que detectar. Luego pasamos al modelo predictivo: por que elegimos XGBoost, como lo tuneamos y que resultados obtuvimos. Despues traduciremos esos resultados a valor de negocio con la curva de lift y recomendaciones accionables. Veremos la arquitectura del pipeline con Airflow y Docker. Y finalmente, conclusiones y siguientes pasos."

---

## Diapositiva 3 — Seccion: Contexto del Problema (15 segundos)

**Lo que se ve:** Titulo de seccion sobre fondo oscuro.

**Que decir:**

> "Empecemos por entender el problema que estamos resolviendo."

---

## Diapositiva 4 — El Problema: Churn en Telecomunicaciones (2 minutos)

**Lo que se ve:** Texto explicativo a la izquierda, 3 tarjetas con metricas clave a la derecha (100K, 100, 50/50).

**Que decir:**

> "El churn, o la tasa de abandono de clientes, es uno de los problemas mas costosos en telecomunicaciones. Se estima que captar un nuevo cliente cuesta entre 5 y 7 veces mas que retener uno existente. En telco, la tasa de churn mensual tipica esta entre el 1 y el 3%, y cada punto porcentual de reduccion puede representar millones de euros en ingresos retenidos.
>
> Nuestro objetivo es construir un modelo que identifique a los clientes con mayor probabilidad de irse, ANTES de que se vayan, para poder actuar con campanas de retencion dirigidas.
>
> El dataset que nos proporcionaron tiene 100.000 clientes, 100 variables que incluyen datos de uso, facturacion, llamadas a soporte, antiguedad del equipo, etc. Un detalle importante: el dataset esta balanceado artificialmente al 50/50 entre churn y no-churn. En la realidad, el churn suele ser del 1-3%, asi que este balance es artificial. Esto lo tendremos en cuenta a la hora de interpretar metricas como accuracy."

**Posibles preguntas:**
- *"Por que importa que este balanceado artificialmente?"* → Porque accuracy sera engañosamente alta. En produccion con datos reales (2% churn) necesitariamos ajustar con class_weight o SMOTE.

---

## Diapositiva 5 — Seccion: Analisis Exploratorio (15 segundos)

> "Pasemos al analisis exploratorio, que fue la parte mas reveladora del proyecto."

---

## Diapositiva 6 — EDA: Hallazgos Principales (2.5 minutos)

**Lo que se ve:** 4 tarjetas con hallazgos clave: balance artificial, nulos camuflados, multicolinealidad, valores negativos.

**Que decir:**

> "El EDA revelo que el dataset contenia varias trampas intencionadas. Vamos una por una.
>
> **Balance artificial 50/50**: como ya mencione, tener exactamente 50% churn es poco realista. Esto nos dice que el dataset fue construido para propositos de testing. La ventaja es que no necesitamos tecnicas de oversampling como SMOTE, pero debemos recordar que en produccion las proporciones seran muy diferentes.
>
> **Nulos camuflados**: las columnas kid0_2 hasta kid16_17 usaban la letra 'U' como valor en mas del 90% de los casos. Otros campos usaban 'Z'. Estos no son valores reales, son nulos disfrazados que un pipeline automatico no detectaria con isnull(). Los reemplazamos por NaN antes de cualquier analisis.
>
> **Multicolinealidad extrema**: encontramos pares de variables con correlacion de 1.000, como rev_Mean y totrev, que son literalmente la misma informacion. Tener ambas es redundante y puede causar inestabilidad en modelos lineales. Eliminamos la del par que tenia menor correlacion con churn, usando un umbral de r > 0.98.
>
> **Valores negativos en eqpdays**: eqpdays representa los dias de antiguedad del equipo del cliente. Encontramos valores negativos, lo cual es fisicamente imposible. Los convertimos a NaN y se imputaron con la mediana."

**Posibles preguntas:**
- *"Como detectaste los nulos camuflados?"* → Analizando las distribuciones de frecuencia de las categoricas. Cuando una categoria tiene >90% del mismo valor, es sospechoso.
- *"Por que mediana y no media para imputar?"* → La mediana es robusta a outliers. En variables sesgadas (como ingresos), la media estaria distorsionada.

---

## Diapositiva 7 — Trampas Detectadas en el Dataset (2 minutos)

**Lo que se ve:** 5 filas con badges de severidad (CRITICA, MEDIA, VALIDADA, ALTA).

**Que decir:**

> "Ademas de los hallazgos del EDA basico, detectamos trampas adicionales.
>
> La mas **critica**: Customer_ID esta ordenado y correlacionado con churn. Si no lo eliminamos, el modelo aprenderia el orden de los IDs en vez de patrones reales. Esto seria data leakage puro.
>
> **Varianza casi nula**: variables donde mas del 99% de las filas tienen el mismo valor. No aportan capacidad discriminativa y solo añaden ruido.
>
> **Dummies sin señal**: variables binarias donde la tasa de churn es identica para 0 y para 1. Es decir, no tienen correlacion alguna con el target.
>
> Un caso interesante: **change_mou**, que es el cambio en minutos de uso. Es el predictor numero 1 segun SHAP, pero nos preguntamos si podria ser leakage: si representa el cambio DESPUES de que el cliente decide irse, seria trampa. Hicimos un estudio de ablacion que veremos mas adelante, y confirmamos que NO es leakage.
>
> Y finalmente, eliminamos todas las columnas con mas del 85% de nulos. Imputar mas del 85% seria basicamente inventar datos."

---

## Diapositiva 8 — Feature Engineering: 8 Variables Nuevas (2 minutos)

**Lo que se ve:** Tabla con las 8 features, su descripcion y logica de calculo.

**Que decir:**

> "Tras la limpieza, creamos 8 features nuevas, todas con interpretacion de negocio clara.
>
> **rev_per_minute**: ingreso por minuto de uso. Mide cuanto paga el cliente por cada minuto que habla. Un valor alto puede indicar un plan caro relativo al uso.
>
> **call_fail_rate**: porcentaje de llamadas que fallan (caidas + bloqueadas sobre intentos totales). Si un cliente sufre muchos fallos de red, su experiencia se degrada y es mas probable que se vaya.
>
> **care_intensity**: ratio de llamadas al servicio de atencion al cliente sobre el uso total. Un cliente que llama mucho a soporte pero usa poco el servicio probablemente esta frustrado.
>
> **old_device**: flag binaria, 1 si el equipo tiene mas de un año. Los clientes con equipos viejos suelen estar menos comprometidos o esperando a que expire el contrato.
>
> **usage_drop_severe**: 1 si el cambio en minutos de uso es menor a -50. Una caida severa es una señal de desenganche.
>
> **rev_vs_avg3**: ratio entre la facturacion actual y la media de los ultimos 3 meses. Detecta cambios bruscos en el patron de gasto.
>
> **new_customer**: 1 si el cliente tiene menos de 6 meses. Los clientes nuevos son mas volatiles.
>
> **engagement_score**: score compuesto que combina el ranking de uso, llamadas completadas y recibidas. Da una vision holistica del engagement."

**Posibles preguntas:**
- *"Por que no usaste PCA o seleccion automatica de features?"* → Porque el objetivo era mantener la interpretabilidad. SHAP sobre features interpretables da insights accionables para el negocio.

---

## Diapositiva 9 — Seccion: Modelo Predictivo (15 segundos)

> "Pasemos a la parte de modelado."

---

## Diapositiva 10 — Pipeline de Preprocesamiento (1.5 minutos)

**Lo que se ve:** 5 tarjetas en flujo horizontal: Limpieza → FE → LabelEncoder → Imputacion → Split.

**Que decir:**

> "Antes de entrenar, el preprocesamiento sigue estos 5 pasos en orden.
>
> Primero, la limpieza de todas las trampas que vimos. Despues, el feature engineering con las 8 variables nuevas. Luego, LabelEncoder para convertir las categoricas a numerico — elegimos LabelEncoder sobre OneHotEncoder porque con arboles de decision (XGBoost) funciona igual de bien y evita la explosion de dimensionalidad.
>
> La imputacion de nulos numericos se hace con la mediana, que es robusta a outliers. Y finalmente, un split estratificado 70/15/15: train, validation y test. El validation set se usa durante el entrenamiento para early stopping y sanity check, y el test set se reserva exclusivamente para la evaluacion final.
>
> El resultado son 100.000 filas por 88 columnas limpias, sin leakage."

**Posibles preguntas:**
- *"Por que 70/15/15 y no 80/20?"* → Necesitamos un validation set separado para tuning y monitoring durante el entrenamiento, ademas del test set intocable.
- *"Por que no StandardScaler?"* → XGBoost es invariante a escala porque usa arboles. El scaler se guarda en los artefactos por si usamos Logistic Regression.

---

## Diapositiva 11 — Benchmarking: 5 Modelos Evaluados (2 minutos)

**Lo que se ve:** Grafico de barras horizontales con AUC-ROC de 5 modelos + panel explicativo "Por que XGBoost".

**Que decir:**

> "Probamos 5 modelos en orden de complejidad creciente.
>
> **Logistic Regression** como baseline: AUC de 0.626. Es el piso minimo que cualquier modelo deberia superar. Lo mantenemos como referencia interpretable.
>
> **Decision Tree**: 0.638. Mejora ligeramente pero sufre de overfitting.
>
> **Random Forest**: 0.671. El ensemble ya mejora significativamente al arbol individual.
>
> **XGBoost y LightGBM empatan en 0.694**. Ambos son gradient boosting, pero elegimos XGBoost por varias razones: tiene regularizacion L1/L2 nativa que ayuda a controlar overfitting, tiene mejor soporte para SHAP con TreeExplainer optimizado, tiene un ecosistema mas maduro y documentado, y el parametro scale_pos_weight nos da control directo sobre el desbalance de clases cuando lo necesitemos en produccion.
>
> La Logistic Regression se mantiene como baseline interpretable. La diferencia de casi 7 puntos de AUC entre LR y XGBoost justifica la complejidad adicional."

**Posibles preguntas:**
- *"Probaste redes neuronales?"* → No, porque con datos tabulares de 100K filas, los gradient boosting trees son consistentemente superiores segun la literatura. Ademas, perderiamos interpretabilidad con SHAP.

---

## Diapositiva 12 — Hyperparameter Tuning (1.5 minutos)

**Lo que se ve:** Panel izquierdo con parametros de configuracion, panel derecho con razonamiento.

**Que decir:**

> "Para el tuning usamos RandomizedSearchCV con 30 iteraciones y 3-fold cross-validation.
>
> ¿Por que RandomizedSearch y no GridSearch? Con 11 hiperparametros, el grid completo tendria miles de combinaciones. Bergstra y Bengio demostraron en 2012 que el random search es mas eficiente porque explora el espacio de forma mas uniforme, especialmente cuando no todos los parametros son igualmente importantes.
>
> Usamos 3-fold porque con 100.000 filas, cada fold tiene unas 67.000 muestras, que es mas que suficiente para una estimacion estable. 5-fold triplicaria el tiempo sin mejorar significativamente la estimacion.
>
> La metrica objetivo es AUC-ROC porque es threshold-independent. Optimizar AUC nos da el mejor modelo posible sin comprometernos a un punto de corte fijo, que luego ajustaremos segun el coste de negocio.
>
> Los parametros de regularizacion (reg_alpha y reg_lambda) son clave: con 88 features, queremos controlar el overfitting."

---

## Diapositiva 13 — Metricas Finales en Test Set (2.5 minutos)

**Lo que se ve:** 4 tarjetas grandes con AUC-ROC 0.7023, F1 0.6448, Recall 0.6476, Precision 0.6420 + confusion matrix + interpretacion.

**Que decir:**

> "Estos son los resultados finales sobre el test set, que el modelo nunca vio durante el entrenamiento.
>
> **AUC-ROC de 0.7023**: esto indica una capacidad discriminativa moderada-buena. Puede parecer modesto, pero hay que tener en cuenta que estamos trabajando con un dataset de 100 columnas llenas de trampas y ruido. Un AUC de 0.70 sobre datos limpios y con features interpretables es un resultado solido.
>
> **F1 Score de 0.6448**: equilibra precision y recall. En nuestro caso, ambas estan bastante parejas, lo cual es bueno.
>
> **Recall de 0.6476**: de cada 100 clientes que realmente van a hacer churn, nuestro modelo identifica 65. Los otros 35 se nos escapan. En negocio, esto significa que capturamos casi dos tercios de los churners.
>
> **Precision de 0.6420**: de cada 100 clientes que el modelo marca como "va a hacer churn", 64 realmente lo hacen. Los otros 36 son falsos positivos, clientes que recibirian una oferta de retencion innecesaria.
>
> Si miramos la confusion matrix: de 7.500 clientes en test que realmente hicieron churn, identificamos 4.856 (los true positives). Y de los 7.500 que no hicieron churn, clasificamos correctamente 4.718.
>
> Un punto importante: el threshold de 0.50 es ajustable. Dependiendo de si el negocio prefiere maximizar recall (no perder ningun churner, aunque contactemos a algunos que no lo son) o precision (contactar solo a los seguros), podemos moverlo."

---

## Diapositiva 14 — Entendiendo las Metricas (1.5 minutos)

**Lo que se ve:** 4 filas explicando AUC-ROC, Precision, Recall, F1.

**Que decir:**

> "Vamos a asegurarnos de que entendemos que mide cada metrica y por que es relevante.
>
> **AUC-ROC** es el area bajo la curva que grafica la tasa de verdaderos positivos contra falsos positivos a todos los thresholds posibles. Un valor de 1.0 seria un modelo perfecto, 0.5 seria tirar una moneda. Usamos esta como metrica principal porque no depende de un punto de corte arbitrario.
>
> **Precision** responde a la pregunta '¿de los que marqué como churn, cuantos realmente lo son?'. Alta precision significa menos gastos en retencion de clientes que no iban a irse.
>
> **Recall** responde '¿de los churners reales, cuantos estoy detectando?'. Alto recall significa menos ingresos perdidos por clientes no identificados.
>
> **F1-Score** es la media armonica de precision y recall. Es util cuando ambos costes (FP y FN) son similares. Si no, hay que priorizar una u otra segun el contexto de negocio."

---

## Diapositiva 15 — SHAP: Interpretabilidad del Modelo (2 minutos)

**Lo que se ve:** Ranking de top 10 features por importancia SHAP (barras) + panel de interpretacion.

**Que decir:**

> "SHAP, o SHapley Additive exPlanations, nos permite entender por que el modelo toma cada decision. Esta basado en la teoria de juegos cooperativos de Shapley: cada feature recibe un valor que representa su contribucion marginal a la prediccion.
>
> El predictor mas fuerte es **change_mou**, el cambio en minutos de uso. Tiene todo el sentido: un cliente que reduce drasticamente su uso probablemente esta considerando cambiar de proveedor.
>
> **mou_Mean** (minutos de uso promedio) y **eqpdays** (antiguedad del equipo) son el segundo y tercer predictor. Clientes con poco uso y equipos viejos tienen mayor riesgo.
>
> **months** (antiguedad del cliente) aparece cuarto: clientes nuevos, con menos de 6 meses, son significativamente mas volatiles.
>
> Un dato interesante: **call_fail_rate**, una de nuestras features nuevas de feature engineering, aparece en el top 8. Esto valida que la calidad de servicio percibida impacta directamente en churn.
>
> Lo mas poderoso de SHAP es que no solo da importancias globales: podemos generar explicaciones por cliente individual. 'Este cliente tiene riesgo 78% porque su uso cayo un 60%, tiene el equipo mas viejo, y ha llamado 5 veces al servicio tecnico.' Eso es lo que el equipo de retencion necesita para personalizar la intervencion."

**Posibles preguntas:**
- *"Cual es la diferencia entre SHAP y feature importance de XGBoost?"* → La feature importance nativa de XGBoost esta basada en gain y puede ser inconsistente. SHAP satisface propiedades axiomaticas (eficiencia, simetria, dummy) que la hacen teoricamente mas robusta.

---

## Diapositiva 16 — Seccion: Valor de Negocio (15 segundos)

> "Ahora traduzcamos estos numeros a impacto de negocio."

---

## Diapositiva 17 — Curva de Lift: Impacto en Negocio (2 minutos)

**Lo que se ve:** Grafico de linea con curva de lift vs aleatorio + panel de impacto practico.

**Que decir:**

> "La curva de lift es probablemente la metrica mas relevante para el negocio. Responde a una pregunta directa: si solo puedo contactar a un porcentaje de mis clientes, ¿cuantos churners voy a capturar?
>
> La linea azul es nuestro modelo y la gris es el baseline aleatorio. Si contactamos al azar al 10% de los clientes, capturaremos el 10% de los churners. Con nuestro modelo, contactando al top 10% de clientes por riesgo segun el modelo, capturamos un 16% de los churners. Eso es un **lift de 1.62x**, es decir, un 62% mas eficiente que contactar al azar.
>
> En la practica, si una telco tiene 10 millones de clientes y un churn mensual del 2%, eso son 200.000 churners al mes. Contactar al azar al 10% (1 millon de clientes) capturaria 20.000. Con nuestro modelo, capturariamos 32.000 con el mismo esfuerzo. Son 12.000 clientes mas retenidos por mes.
>
> Si el ARPU (ingreso medio por usuario) es de 30 euros al mes, esos 12.000 clientes extra representan 360.000 euros mensuales en ingresos retenidos. Descontando el coste de las campanas de retencion, el ROI suele ser positivo contactando al top 20-30%."

---

## Diapositiva 18 — Estudio de Ablacion: change_mou (1.5 minutos)

**Lo que se ve:** Explicacion del problema, tabla comparativa con/sin change_mou, conclusion con check verde.

**Que decir:**

> "Este es un punto que quiero destacar porque demuestra rigor analitico. change_mou es el predictor numero 1, pero nos planteamos una pregunta critica: ¿podria ser leakage?
>
> El argumento seria: si change_mou representa el cambio en uso DESPUES de que el cliente ya decidio irse, entonces estariamos haciendo trampa. El modelo estaria usando informacion del futuro para predecir algo que ya paso.
>
> Para verificarlo, hicimos un estudio de ablacion: entrenamos el modelo completo quitando change_mou y comparamos las metricas. El AUC cayo de 0.7023 a aproximadamente 0.67, un delta de solo 3 puntos porcentuales. El F1 cayo similarmente unos 2.5 puntos.
>
> Nuestra regla: si el delta fuera mayor a 5 puntos porcentuales, seria sospechoso de leakage. Con un delta de 3pp, concluimos que change_mou es una señal legitima de comportamiento previo al churn, no informacion del futuro. El cliente empieza a usar menos el servicio antes de irse, y eso es lo que estamos capturando."

---

## Diapositiva 19 — Recomendaciones de Negocio (2 minutos)

**Lo que se ve:** 5 recomendaciones con iconos.

**Que decir:**

> "Basandonos en los insights del modelo y SHAP, tenemos 5 recomendaciones accionables.
>
> **Uno: campanas de retencion focalizadas**. Usar el modelo para priorizar al top 20-30% de clientes por riesgo. Con el lift de 1.5x, cada euro invertido en retencion rinde un 50% mas que contactar al azar.
>
> **Dos: programa de renovacion de equipos**. eqpdays es un predictor fuerte. Los clientes con equipos viejos (mas de un año) tienen mayor probabilidad de churn. Un programa de upgrade subsidiado podria retenerlos.
>
> **Tres: alertas de calidad de red**. call_fail_rate es un predictor significativo. Si monitorizamos la calidad de red por zona y actuamos proactivamente ante degradaciones, podemos prevenir churn antes de que el cliente llame a quejarse.
>
> **Cuatro: onboarding reforzado**. Los clientes nuevos (menos de 6 meses) son los mas volatiles. Un programa de bienvenida con incentivos en los primeros 90 dias puede marcar la diferencia.
>
> **Cinco: scoring semanal automatizado**. Ejecutar el pipeline periodicamente, alimentar el CRM con scores de riesgo actualizados, y generar listas de accion para el equipo de retencion."

---

## Diapositiva 20 — Seccion: Arquitectura del Pipeline (15 segundos)

> "Veamos como se lleva todo esto a un entorno de produccion."

---

## Diapositiva 21 — Apache Airflow: DAG de 3 Tasks (2 minutos)

**Lo que se ve:** 3 tarjetas en flujo (data_preparation → train_model → evaluate_model) con tiempos de ejecucion.

**Que decir:**

> "El pipeline esta orquestado con Apache Airflow en un DAG de exactamente 3 tasks, como pide la prueba.
>
> **Task 1: data_preparation**, que tarda unos 16 segundos. Carga el dataset original, ejecuta toda la limpieza de trampas, el feature engineering, el encoding y el split. Guarda los datasets procesados como CSV y los artefactos de preprocesamiento como pickle.
>
> **Task 2: train_model**, unos 5 segundos. Carga los datos procesados, entrena el XGBoost con los hiperparametros optimizados, hace un sanity check en el validation set, y guarda el modelo completo como pickle. El pickle incluye no solo el modelo sino tambien los artefactos de preprocesamiento (encoders, imputer, scaler) para que sea auto-contenido.
>
> **Task 3: evaluate_model**, menos de 1 segundo. Carga el pickle y el test set, genera predicciones, calcula todas las metricas (AUC-ROC, F1, confusion matrix, lift curve), las imprime en los logs de Airflow, y guarda un JSON con el reporte completo. Ademas tiene un quality gate: si el AUC cae por debajo de 0.55, el task falla.
>
> Usamos PythonOperator porque cada task importa y ejecuta la funcion main() del script correspondiente. Si un paso falla, el DAG se detiene y no ejecuta los siguientes."

---

## Diapositiva 22 — Infraestructura: Docker Compose (1.5 minutos)

**Lo que se ve:** 4 servicios de Docker, panel de decisiones de diseño, comandos.

**Que decir:**

> "Todo corre sobre Docker Compose con 4 servicios: PostgreSQL como base de datos de metadata de Airflow, el Webserver para la interfaz grafica en el puerto 8080, el Scheduler que ejecuta los DAGs, y el Triggerer para operadores diferibles.
>
> Decisiones de diseño importantes: elegimos **LocalExecutor** en vez de CeleryExecutor porque para un solo DAG secuencial no necesitamos la complejidad de Celery + Redis. Esto reduce el consumo de RAM de 8GB a unos 4GB.
>
> La imagen es **custom**: la imagen base de Airflow no incluye xgboost, scikit-learn ni las otras dependencias de ML. Nuestro Dockerfile hereda de la imagen oficial y ejecuta pip install de nuestro requirements.txt.
>
> Los **volumenes** montan las carpetas scripts/, data/ y models/ directamente en el contenedor. Esto significa que los cambios en los scripts se reflejan inmediatamente sin reconstruir la imagen.
>
> Para ejecutar todo, son literalmente 3 comandos: build, init, up. Y la interfaz queda disponible en localhost:8080 con user/pass airflow/airflow."

---

## Diapositiva 23 — Seccion: Conclusiones (15 segundos)

> "Para cerrar, un resumen de lo construido y los siguientes pasos."

---

## Diapositiva 24 — Resumen y Siguientes Pasos (2 minutos)

**Lo que se ve:** Checklist de lo construido (izquierda) + lista de siguientes pasos (derecha).

**Que decir:**

> "Resumiendo lo que hemos construido: un EDA profundo con deteccion de mas de 13 trampas, 8 features nuevas con logica de negocio, un XGBoost tuneado con AUC de 0.70 y lift de 1.62x, interpretabilidad con SHAP, un estudio de ablacion para validar que no hay leakage, 3 scripts modulares y testeados end-to-end, y un DAG de Airflow con Docker listo para desplegar.
>
> Para **siguientes pasos**, hay varias extensiones que aportarian valor:
>
> **MLflow** para experiment tracking: versionar modelos, comparar runs, registrar artefactos con trazabilidad completa.
>
> **PostgreSQL** para persistir los datos procesados en vez de CSVs, lo cual es mas robusto y permite queries SQL.
>
> **Prometheus + Grafana** para monitorizar el drift del modelo en produccion: si la distribucion de los datos cambia, las predicciones se degradan y hay que reentrenar.
>
> **Feature store** para centralizar features reutilizables entre equipos y modelos.
>
> Y cuando trabajemos con **datos reales desbalanceados** (churn del 2%), tendremos que adaptar con class_weight o SMOTE y recalibrar los thresholds."

---

## Diapositiva 25 — Gracias / Q&A (abierto)

**Lo que se ve:** "Gracias", "Preguntas y Respuestas", contacto.

**Que decir:**

> "Con esto concluyo la presentacion. Quedo abierto a cualquier pregunta sobre el EDA, el modelo, la arquitectura, o cualquier decision tecnica que quieran profundizar. Gracias."

---

## Apendice: Preguntas Frecuentes Anticipadas

### Sobre los datos
- **"¿Que harias diferente con datos reales?"** → Primero, SMOTE o class_weight para el desbalance. Segundo, validacion temporal (train en meses anteriores, test en el ultimo mes). Tercero, analisis de drift periodico.
- **"¿Por que no eliminaste mas variables?"** → Mantuvimos todas las que pasaron los filtros de varianza, multicolinealidad y señal. XGBoost maneja bien features ruidosas gracias a su regularizacion.

### Sobre el modelo
- **"¿Por que no deep learning?"** → Para datos tabulares de este tamaño, los gradient boosting trees son state-of-the-art segun benchmarks recientes (Grinsztajn et al., 2022). Ademas, SHAP TreeExplainer es exacto, mientras que SHAP en redes neuronales es aproximado.
- **"¿El AUC de 0.70 es suficiente?"** → Depende del contexto. Para un dataset con 100 features ruidosas y trampas, es solido. En produccion, con feature engineering adicional, datos temporales, y variables de contrato, podria mejorar a 0.75-0.80.
- **"¿Que pasa si el modelo se degrada con el tiempo?"** → Model drift. Por eso es importante monitorizar metricas en produccion y reentrenar periodicamente (propuesta de Prometheus + Grafana).

### Sobre la arquitectura
- **"¿Por que Airflow y no Luigi/Prefect/Dagster?"** → Airflow es el estandar de la industria, tiene la mayor comunidad, integracion nativa con la mayoria de clouds, y es lo que pide la prueba tecnica.
- **"¿Como escalarias esto?"** → CeleryExecutor con Redis para paralelismo, Kubernetes para auto-scaling, y separar el entrenamiento en una maquina con GPU si es necesario.
- **"¿Por que PythonOperator y no BashOperator?"** → PythonOperator nos da mejor control de errores, logging integrado, y paso de contexto entre tasks via XCom si lo necesitamos.
