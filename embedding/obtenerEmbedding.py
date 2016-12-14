
from codecs import open, BOM_UTF8
import re

# Cargo lexicon
print 'Cargando lexicon...'
lexicon = []
for linea in open('../lexicon/lexicon.txt', encoding="latin-1"):
    r = linea.split()
    if len(r) > 0:
        lexicon.append(r[0])


print 'Cargando embedding completo...'
# Cargo embedding pre-entrenado
embedding_completo = {}
palabras_embedding_completo = []
palabras_corruptas = 0
for linea in open('./eswiki.0.3.vectors.150.txt', encoding="latin-1"):
    r = linea.split()
    vector = r[1:]
    embedding_completo[r[0]] = vector
    palabras_embedding_completo.append(r[0])

    # try:
    #     linea.decode('utf-8', 'strict')
    #     r = linea.split()
    #     vector = r[1:]
    #     embedding_completo[r[0]] = vector
    #     palabras_embedding_completo.append(r[0])
    # except:
    #     r = linea.split()
    #     print r[0]
    #     palabras_corruptas += 1

print 'Se encontraron ' + str(palabras_corruptas) + ' palabras corruptas'

print 'Filtrando palabras...'
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
print 'Persistiendo resultados...'
open("embedding.txt", "w").write(BOM_UTF8 + "\n".join(embedding).encode("latin-1"))
open("faltantes.txt", "w").write(BOM_UTF8 + "\n".join(no_encontradas).encode("latin-1"))
open("lexicon.txt", "w").write(BOM_UTF8 + "\n".join(nuevo_lexicon).encode("latin-1"))

open("palabras_embedding.txt", "w").write(BOM_UTF8 + "\n".join(palabras_embedding_completo).encode("latin-1"))

print 'Se obtuvo el embedding de ' + str(len(embedding)) + ' palabras'
print 'Se perdieron ' + str(len(no_encontradas)) + ' palabras en el proceso'
