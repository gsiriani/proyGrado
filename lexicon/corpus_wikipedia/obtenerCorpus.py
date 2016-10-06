import sys
import json
sys.path.append('.../myfreeling/APIs/python')
import freeling
import random
import io
from codecs import open, BOM_UTF8
from glob import glob

# Variables de control
# Seteadas para cubrir todo el corpus
words_in_corpus = 130000000
max_words = 55000
start=0

print("Obteniendo Corpus...")

# Inicializo diccionarios
lexicon = {}
numeros = {}
fechas = {}
desconocidos = {}

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

def agregar_a_fechas(palabra):        
    if not fechas.has_key(palabra):
        fechas[palabra] = 1

def agregar_a_desconocidos(palabra, tag):        
    if not desconocidos.has_key(palabra):
        desconocidos[palabra] = tag

def has_num(palabra):
  return any(char.isdigit() for char in palabra)

# Configuracion de freeling
# -------------------------
FREELINGDIR = "/usr/local";
DATA = FREELINGDIR+"/share/freeling/"
LANG = "es"

freeling.util_init_locale("default")

# Se crean opciones para analizador maco
op= freeling.maco_options("es")
op.set_data_files( "", 
                   DATA + "common/punct.dat",
                   DATA + LANG + "/dicc.src",
                   DATA + LANG + "/afixos.dat",
                   "",
                   DATA + LANG + "/locucions.dat", 
                   DATA + LANG + "/np.dat",
                   DATA + LANG + "/quantities.dat",
                   DATA + LANG + "/probabilitats.dat")

# Se crea el analizador maco con las opciones precreadas
morfo = freeling.maco(op)
# Se setean los analisis requeridos. Solamente se usa deteccion de numeros y de fechas 
morfo.set_active_options (False, # UserMap
                         False, # NumbersDetection,
                         False, #  PunctuationDetection,
                         False, #  DatesDetection,  --> Setear a True para considerar fechas
                         False, #  DictionarySearch,
                         False, #  AffixAnalysis,
                         False, #  CompoundAnalysis,
                         False, #  RetokContractions,
                         False, #  MultiwordsDetection,
                         False, #  NERecognition,
                         False, #  QuantitiesDetection,
                         False) #  ProbabilityAssignment

# Se crean tokenizador y splitter
tk = freeling.tokenizer(DATA+LANG+"/tokenizer.dat")
sp = freeling.splitter(DATA+LANG+"/splitter.dat")


# Estructura principal
# --------------------
i = 0 # Contador de palabras leidas
motivo_fin = "Se han parseado todos los archivos"
# Recorro los archivos del corpus
for f in glob("raw.es/*")[start:]:
    # Recorro las lineas del archivo
    for line in open(f, encoding="latin-1"):
        if line == "\n" or line.startswith((
          "<doc", "</doc>", "ENDOFARTICLE", "REDIRECT",
          "Acontecimientos", 
          "Fallecimientos", 
          "Nacimientos"," Acontecimientos", 
          " Fallecimientos", 
          " Nacimientos")):
            continue

        # Si ya tengo la cantidad de palabras que quiero, corto
        # el loop
        if i >= words_in_corpus:
            motivo_fin = "Supere la cantidad de palabras previstas"
            break

        # Divido la linea en tokens (palabras)
        palabras_linea = tk.tokenize(line.lower())

        # Agrupo en oraciones para analizar
        for oracion in sp.split(palabras_linea):
          # # Analizo la oracion          
          # oracion = morfo.analyze(oracion)
          palabras = oracion.get_words()

        	# Recorro las palabras de la oracion
          for palabra in palabras:	
	        	# Actualizo la cantidad de palabras leidas
            i += 1
        		# Descarto las que tienen numero
            if not has_num(palabra.get_form()):
        			# Agrego palabra al diccionario
              agregar_a_lexicon(palabra.get_form())
            else:
              agregar_a_numeros(palabra.get_form())
 
print("Ordenando palabras...")

# Grabo los resultados en archivos de texto
# ------------------------------------------

# Palabras mas frecuentes
# - - - - - - - - - - - -
# Ordeno segun la frecuencia y me quedo con la cantidad deseada        
top = []  
for (w,freq) in lexicon.items():    
    top.append((freq, w))
 
top = sorted(top, reverse=True)[:max_words] # top max_words
resultado = [w for (_,w) in top] 	# Almaceno unicamente las palabras
# resultado = [str(f) + ' ' + w for (f,w) in top] # Almaceno palabras y cantidad de usos

# Guardo las palabras en un archivo de texto
open("lexicon_wiki-frecuentes.txt", "w").write(BOM_UTF8 + "\n".join(resultado).encode("utf-8"))

# Todas las palabras consideradas
# - - - - - - - - - - - - - - - -
total = []
for (w,freq) in lexicon.items():    
    total.append((freq, w)) 
total = sorted(total, reverse=True)
# resultado = [w for (_,w) in top] 	# Almaceno unicamente las palabras
resultado_tot = [str(f) + ' ' + w for (f,w) in total] # Almaceno palabras y cantidad de usos
# Guardo las palabras en un archivo de texto
open("lexicon_wiki-total.txt", "w").write(BOM_UTF8 + "\n".join(resultado_tot).encode("utf-8")) 

# Palabras identificadas como numero
# - - - - - - - - - - - - - - - - -
numeros_aux = [w for (w,_) in numeros.items()]
open("descarte/lexicon_wiki-numeros.txt", "w").write(BOM_UTF8 + "\n".join(numeros_aux).encode("utf-8"))

# Palabras identificadas como fecha
# - - - - - - - - - - - - - - - - -
fechas_aux = [w for (w,_) in fechas.items()]
open("descarte/lexicon_wiki-fechas.txt", "w").write(BOM_UTF8 + "\n".join(fechas_aux).encode("utf-8"))

# Palabras con tag desconocida
# - - - - - - - - - - - - - - 
desconocidos_aux = [str(w) + ' ' + str(t) for (w,t) in desconocidos.items()]
open("descarte/lexicon_wiki-desconocidos.txt", "w").write(BOM_UTF8 + "\n".join(desconocidos_aux).encode("utf-8"))


# Calculo porcentaje de cobertura
# -------------------------------
porcentaje = (100*max_words)/len(lexicon)
print("La cantidad total de palabras parseadas es: " + str(len(lexicon)))
print("El porcentaje de cobertura es: " + str(porcentaje))
print(motivo_fin)
