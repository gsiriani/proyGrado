import sys
import json
import random
import io
from codecs import open, BOM_UTF8
from glob import glob

print("Obteniendo Corpus...")

# Inicializo diccionarios
lexicon = {}
numeros = {}

# Funciones auxiliares
# --------------------
def agregar_a_lexicon(palabra):        
    if lexicon.has_key(palabra):
        lexicon[palabra] += 1
    else:
        lexicon[palabra] = 1

def agregar_a_numeros(palabra):        
    if not numeros.has_key(palabra):
        numeros[palabra] = 1


# Estructura principal
# --------------------
i = 0 # Contador de palabras leidas
# Recorro las oraciones (lineas) del archivo
for line in open('ancora_oraciones.txt', encoding="utf-8"):
  	# Recorro las palabras de la oracion
    for palabra in line[:-3].split(' '):	# el [:-3] es para deshacerme del salto de linea al final de cada oracion
      	# Actualizo la cantidad de palabras leidas
        i += 1
        if palabra.isdigit() and len(palabra) > 0:
            agregar_a_numeros(palabra)
        else:
            agregar_a_lexicon(palabra.lower())
 
print("Ordenando palabras...")

# Grabo los resultados en archivos de texto
# ------------------------------------------

# Palabras mas frecuentes
# - - - - - - - - - - - -
# Ordeno segun la frecuencia y me quedo con la cantidad deseada        
top = []  
for (w,freq) in lexicon.items():    
    top.append((freq, w))
 
top = sorted(top, reverse=True)# palabras ordenadas por frecuencia
resultado = [w for (_,w) in top] 	# Almaceno unicamente las palabras
# resultado = [str(f) + ' ' + w for (f,w) in top] # Almaceno palabras y cantidad de usos

# Guardo las palabras en un archivo de texto
open("lexicon_ancora.txt", "w").write(BOM_UTF8 + "\n".join(resultado).encode("utf-8"))


# Palabras identificadas como numero
# - - - - - - - - - - - - - - - - -
numeros_aux = [w for (w,_) in numeros.items()]
open("descarte/lexicon_ancora-numeros.txt", "w").write(BOM_UTF8 + "\n".join(numeros_aux).encode("utf-8"))


# Calculo porcentaje de cobertura
# -------------------------------
print("La cantidad total de palabras parseadas es: " + str(len(lexicon)))