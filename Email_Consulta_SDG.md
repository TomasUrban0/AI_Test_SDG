Buenas tardes,

Disculpad las molestias, me han surgido algunas dudas respecto a la prueba tecnica que me gustaria aclarar antes de la presentacion:

**Sobre la presentacion:**

- ¿Hay alguna duracion o extension orientativa que tengais en mente para la sesion? Asi puedo ajustar el nivel de detalle.
- ¿Teneis alguna plantilla o modelo base de presentaciones de SDG que deba utilizar, o puedo usar un formato propio?

**Sobre los resultados del modelo:**

El mejor modelo que he obtenido (XGBoost tuneado) alcanza un AUC-ROC de ~0.70 en el holdout test set. Dado que el dataset contiene bastante ruido y varias peculiaridades intencionadas, queria confirmar si este rango de resultados esta dentro de lo esperado o si hay margen para mejorar que deberia explorar antes de la presentacion.

**Observaciones sobre el dataset:**

Durante el analisis exploratorio he detectado lo que interpreto como peculiaridades deliberadas en los datos, entre ellas:

- Las columnas kid0_2 a kid16_17 utilizan "U" como valor en lugar de nulo (>90% de los registros), y otros campos usan "Z" de forma similar.
- La variable eqpdays (dias de antiguedad del equipo) contiene valores negativos, lo cual es fisicamente imposible.
- Varias columnas presentan mas del 85% de valores nulos.

He tratado todas estas anomalias en el preprocesamiento y puedo detallar las decisiones tomadas durante la presentacion.

Quedo a vuestra disposicion para cualquier aclaracion.

Un saludo,
Miguel Ángel Montero
