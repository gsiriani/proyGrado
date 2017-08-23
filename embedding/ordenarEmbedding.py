# -*- coding: utf-8 -*-
path_proyecto = '/home/guille/proyecto/proyGrado'

import sys
sys.path.append(path_proyecto)

from vector_palabras import palabras_comunes
from random import uniform
from script_auxiliares import print_progress
from codecs import open, BOM_UTF8

vector_size = 150

archivo_embedding = path_proyecto + "/embedding/embedding_total.txt"
archivo_lexicon = path_proyecto + "/embedding/lexicon_total.txt"
archivo_embedding_salida = path_proyecto + "/embedding/embedding_ordenado.txt"

print 'Cargando embedding inicial...'
# Cargo embedding inicial
palabras = palabras_comunes(archivo_lexicon) # Indice de cada palabra en el diccionario

cant_palabras = len(palabras)  # Cantidad de palabras consideradas en el diccionario
f = 0
embedding_inicial= [[0] * vector_size] * len(palabras)
for l in open(archivo_embedding):
    print_progress(f, cant_palabras, prefix = 'Progreso:', suffix = 'Completado', bar_length = 50)
    f += 1
    ind_palabra = palabras.obtener_indice(l.split()[0])
    embedding_inicial[ind_palabra] = list([float(x) for x in l.split()[1:]])
    #embedding_inicial.append([float(x) for x in l.split()[1:]])


# Agregamos embedding para 11 signos de puntuacion y PUNCT inicializado como el mismo embedding que ':'
signos_puntuacion = ['.', ',', ';', '(', ')', '¿', '?', '¡', '!', '"', "'", 'PUNCT'] 
indice_punct_base = palabras.obtener_indice(':')
for s in signos_puntuacion:
    ind_palabra = palabras.obtener_indice(s)
    embedding_inicial[ind_palabra] = list(embedding_inicial[indice_punct_base])

# Agregamos embedding para NUM, DATE, OUT y UNK
palabras_extra = ['NUM', 'DATE', 'OUT', 'UNK']
for p in palabras_extra:
    features_aux = []
    for _ in range(vector_size):
        features_aux.append(uniform(-1,1))    
    ind_palabra = palabras.obtener_indice(p)
    embedding_inicial[ind_palabra] = list(features_aux)


print 'Transformo a texto'
t = str(embedding_inicial[0][0])
for e in embedding_inicial[0][1:]:
    t += ' ' + str(e)
for l in embedding_inicial[1:]:
    t += '\n' + str(l[0])
    for e in l[1:]:
        t += ' ' + str(e)



# Guardo embedding ordenado
open(archivo_embedding_salida, "w").write(t)

print 'Fin!'
 