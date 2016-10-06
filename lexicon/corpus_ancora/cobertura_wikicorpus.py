import io
from codecs import open, BOM_UTF8
import re

# Cargo lexicon wikipedia
corpus_wiki = []
for linea in open('../corpus_wikipedia/es-lexicon.txt', encoding="utf-8"):
	r = linea.split()
	if len(r) > 0:
		corpus_wiki.append(r[0])
	

# Cargo lexicon ancora
corpus_ancora = {}
for linea in open('lexicon_ancora.txt', encoding="utf-8"):
	r = linea.split()
	num = int(re.sub("[^0-9]", "", r[0]))
	corpus_ancora[r[1]] = num

aciertos = 0
ocurrencias = 0
cubiertas = 0
faltantes = []
for (palabra, cant) in corpus_ancora.iteritems():
	ocurrencias += cant
	if palabra in corpus_wiki:
		aciertos += 1
		cubiertas += cant
	else:
		faltantes.append(palabra)

print 'Porcentaje de cobertura: ' + str(float(aciertos*100/len(corpus_ancora)))  + '%'
print 'Porcentaje de ocurrencias cubiertas: ' + str(float(cubiertas*100/ocurrencias)) + '%'
open("faltantes.txt", "w").write(BOM_UTF8 + "\n".join(faltantes).encode("utf-8"))

