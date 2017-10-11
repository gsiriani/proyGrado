# Metricas por oración

Obtener en Keras los siguientes resultados para cada tarea en cada modalidad
Tomo todos los casos pertenecientes a una misma oración
Para cada oración obtengo cantidad de palabras totales y cantidad de palabras correctamente etiquetadas.

Obtenemos las siguientes medidas a partir de los resultados obtenidos:
+ Porcentaje de oraciones completamente correctas
+ Correctitud ponderada por largo de la oración
+ Análisis entre largo de oración y cantidad de errores

# NER

+ Porcentaje de casos con entidades con nombre
+ Porcentaje de entidades con nombre reconocidas 
+ Porcentaje de entidades con nombre correctamente etiquetadas

# Chunking

+ Mediciones por chunk (ignorando iobes) (medición especial en keras que chequee el chunk de a 4)
++ Las posibles etiqueta se dividen en chunks de a 4, queremos saber si le pega a ese grupo
+ Analizar caso SIN IOBES (opcional, implica red y caso nuevos)

# POS

+ Conjunto de etiquetas mejor clasificadas
+ Conjunto de etiquetas peor clasificadas
+ Cambios frecuentes de una etiqueta por otra

# SRL

+ Idem chunking


