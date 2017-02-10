#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
from codecs import open, BOM_UTF8
import re

# Funcion auxiliar
def is_punct(palabra):
    return (len(palabra) == 1 and not palabra.isalnum())


ocurrencias_wiki = 0
ocurrencias_ancora = 0


# Cargo lexicon wikipedia
corpus_wiki = {}
total_ocurrencias_wiki = 0
for linea in open('../lexicon/corpus_wikipedia/lexicon_wiki-total.txt', encoding="latin-1"):
    r = linea.split()
    if len(r) > 1:
        num = int(re.sub("[^0-9]", "", r[0]))
        corpus_wiki[r[1]] = num
        total_ocurrencias_wiki += num
        if is_punct(r[1]): # Si es un signo de puntuacion lo considero cubierto
            ocurrencias_wiki += num

# Cargo lexicon ancora
corpus_ancora = {}
total_ocurrencias_ancora = 0
for linea in open('../lexicon/corpus_ancora/lexicon_ancora.txt', encoding="latin-1"):
    r = linea.split()
    num = int(re.sub("[^0-9]", "", r[0]))
    corpus_ancora[r[1]] = num
    total_ocurrencias_ancora += num
    if is_punct(r[1]):  # Si es un signo de puntuacion lo considero cubierto
        ocurrencias_ancora += num

# Cargo lexicon embedding total
lexicon = []
for linea in open('./lexicon_total.txt', encoding="latin-1"):
    r = linea.split()
    lexicon.append(r[0])

for palabra in lexicon:
    if palabra in corpus_wiki and not is_punct(palabra):
        ocurrencias_wiki += corpus_wiki[palabra]
    if palabra in corpus_ancora and not is_punct(palabra):
        ocurrencias_ancora += corpus_ancora[palabra]

# faltantes = {}
# for (palabra, frec) in corpus_ancora.items():
#     if not palabra in lexicon:
#         faltantes[palabra] = frec
#
# # Ordeno segun la frecuencia
# top = []
# for (w, freq) in faltantes.items():
#     top.append((freq, w))
#
# top = sorted(top, reverse=True)
# resultado_tot = [str(f) + ' ' + w for (f,w) in top] # Almaceno palabras y cantidad de usos
# # Guardo las palabras en un archivo de texto
# open("frecuencia_faltantes.txt", "w").write(BOM_UTF8 + "\n".join(resultado_tot).encode("latin-1"))

print 'Porcentaje de ocurrencias cubiertas en Wikipedia: ' + str(float(ocurrencias_wiki * 100 / total_ocurrencias_wiki)) + '%'

print 'Porcentaje de ocurrencias cubiertas en Ancora: ' + str(float(ocurrencias_ancora * 100 / total_ocurrencias_ancora)) + '%'

