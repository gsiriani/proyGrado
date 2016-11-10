
from codecs import open, BOM_UTF8
import re

# Cargo lexicon
lexicon = []
for linea in open('../lexicon/lexicon.txt', encoding="utf-8"):
    r = linea.split()
    if len(r) > 0:
        lexicon.append(r[0])


# Cargo embedding pre-entrenado
embedding_completo = {}
for linea in open('./eswiki.0.3.vectors.150.txt'):
    r = linea.split()
    vector = r[1:]
    embedding_completo[r[0]] = vector

# Conservo las palabras en el lexicon
embedding = []
no_encontradas = []
nuevo_lexicon = []
for palabra in lexicon:
    if palabra in embedding_completo:
        nuevo_lexicon.append(palabra)
        valores = ''
        for num in embedding_completo[palabra]:
            valores += ' ' + str(num)
        embedding.append(palabra + valores)
    else:
        no_encontradas.append(palabra)

# Persisto las palabras
open("embedding.txt", "w").write(BOM_UTF8 + "\n".join(embedding).encode("utf-8"))
open("faltantes.txt", "w").write(BOM_UTF8 + "\n".join(no_encontradas).encode("utf-8"))
open("lexicon.txt", "w").write(BOM_UTF8 + "\n".join(nuevo_lexicon).encode("utf-8"))

print 'Se obtuvo el embedding de ' + str(len(embedding)) + ' palabras'
print 'Se perdieron ' + str(len(no_encontradas)) + ' palabras en el proceso'
