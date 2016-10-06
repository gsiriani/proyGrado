'''
Creo el lexicon final a partir de todas las palabras del corpus de Ancora 
que pertenecen al corpus de Wikipedia.
Ademas calculo el porcentaje de palabras y de cobertura del corpus
'''

import io
from codecs import open, BOM_UTF8
import re

# Cargo lexicon wikipedia
corpus_wiki = []
for linea in open('./corpus_wikipedia/lexicon_wiki-total.txt', encoding="utf-8"):
	r = linea.split()
	if len(r) > 1:
		corpus_wiki.append(r[1])
	

# Cargo lexicon ancora
corpus_ancora = {}
for linea in open('./corpus_ancora/lexicon_ancora.txt', encoding="utf-8"):
	r = linea.split()
	num = int(re.sub("[^0-9]", "", r[0]))
	corpus_ancora[r[1]] = num

# Inicializo el lexicon final
lexicon = []
faltantes = []

aciertos = 0
ocurrencias = 0
cubiertas = 0
for (palabra, cant) in corpus_ancora.iteritems():
	ocurrencias += cant
	if palabra in corpus_wiki:
		aciertos += 1
		cubiertas += cant
		lexicon.append(palabra)
	else:
		faltantes.append(palabra)

print 'Porcentaje de cobertura: ' + str(float(aciertos*100/len(corpus_ancora)))  + '%'
print 'Porcentaje de ocurrencias cubiertas: ' + str(float(cubiertas*100/ocurrencias)) + '%'

# Persisto las palabras
open("lexicon.txt", "w").write(BOM_UTF8 + "\n".join(lexicon).encode("utf-8"))
open("faltantes.txt", "w").write(BOM_UTF8 + "\n".join(faltantes).encode("utf-8"))

