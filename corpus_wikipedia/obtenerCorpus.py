from glob import glob
from codecs import open, BOM_UTF8   

words_in_corpus = 130000000
max_words = 100000
start=0

print "Obteniendo Corpus..."
lexicon = {}

def no_numeros(palabra):
    for p in palabra:
        if (p.isdigit()):
            return False
    return True

def no_simbolos_raros(palabra):
    for p in palabra:
        if (not p.isalnum()):
            return False
    return True

def agregar_a_lexicon(palabra):        
    if lexicon.has_key(palabra):
        lexicon[palabra] += 1
    else:
        lexicon[palabra] = 1

i = 0
motivo_fin = "Se han parseado todos los archivos"
# Recorro los archivos del corpus
for f in glob("raw.es/*")[start:]:
    # Recorro las lineas del archivo
    for line in open(f, encoding="latin-1"):
        if line == "\n" or line.startswith((
          "<doc", "</doc>", "ENDOFARTICLE", "REDIRECT",
          "Acontecimientos", 
          "Fallecimientos", 
          "Nacimientos")):
            continue

        # Si ya tengo la cantidad de palabras que quiero, corto
        # el loop
        if i >= words_in_corpus:
            motivo_fin = "Supere la cantidad de palabras previstas"
            break

        # Divido la linea en palabras
        palabras_linea = line.lower().split()

        # Actualizo la cantidad de palabras leidas
        i += len(palabras_linea)

        # Agrego las palabras al diccionario
        for palabra in palabras_linea:
            # chequeo signo de puntuacion al principio
            inicio = palabra[0]
            if not inicio.isalnum():
                agregar_a_lexicon(inicio)
                palabra = palabra[1:]
            if len(palabra) == 0:
                continue
            # chequeo signo de puntuacion al final                
            final = palabra[-1]
            if not final.isalnum():
                agregar_a_lexicon(final)
                palabra = palabra[:-1]
            # ignoro numeros y palabras de largo 0
            if len(palabra) == 0:
                continue
            if (no_numeros(palabra) and no_simbolos_raros(palabra)):
                agregar_a_lexicon(palabra)
 
print "Ordenando palabras..."

# Ordeno segun la frecuencia y me quedo con la cantidad deseada        
top = []  
for (w,freq) in lexicon.items():    
    top.append((freq, w))
 
top = sorted(top, reverse=True)[:max_words] # top max_words
# resultado = [w for (_,w) in top] 
resultado = [str(f) + ' ' + w for (f,w) in top] 

# Guardo las palabras en un archivo de texto
open("es-lexicon.txt", "w").write(BOM_UTF8 + "\n".join(resultado).encode("utf-8"))


total = []
for (w,freq) in lexicon.items():    
    total.append((freq, w)) 
total = sorted(total, reverse=True)
# resultado = [w for (_,w) in top] 
resultado_tot = [str(f) + ' ' + w for (f,w) in total]
# Guardo las palabras en un archivo de texto
open("es-lexicon-total.txt", "w").write(BOM_UTF8 + "\n".join(resultado_tot).encode("utf-8"))

# Calculo porcentaje de cobertura
porcentaje = (100*max_words)/len(lexicon)
print "La cantidad total de palabras parseadas es: " + str(len(lexicon))
print "El porcentaje de cobertura es: " + str(porcentaje)
print motivo_fin

