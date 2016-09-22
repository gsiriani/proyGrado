import io
from codecs import open, BOM_UTF8

# Cargo lexicon wikipedia
corpus_wiki = []
for palabra in open('../corpus_wikipedia/es-lexicon.txt', encoding="utf-8"):
	corpus_wiki.append(palabra)

# Cargo lexicon ancora
corpus_ancora = []
for palabra in open('lexicon_ancora.txt', encoding="utf-8"):
	corpus_ancora.append(palabra)

aciertos = 0
faltantes = []
for palabra in corpus_ancora:
	if palabra in corpus_wiki:
		aciertos += 1
	else:
		faltantes.append(palabra)

print 'Porcentaje de cobertura: ' + str(aciertos*100/len(corpus_ancora))  + '%'
open("faltantes.txt", "w").write(BOM_UTF8 + "\n".join(faltantes).encode("utf-8"))

