import sys
import json
sys.path.append('.../myfreeling/APIs/python')
import freeling
import random
import io
from codecs import open, BOM_UTF8
from glob import glob

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

def es_fecha(palabra, tk, sp, morfo):
    line = " ".join(palabra.split('_'))

    # Divido la linea en tokens (palabras)
    palabras_linea = tk.tokenize(line.lower())

    # Agrupo en oraciones para analizar
    oracion = sp.split(palabras_linea)[0]
    oracion = morfo.analyze(oracion)
    palabras = oracion.get_words()
    analisis = palabras[0].get_analysis()
    return analisis != () and "W" in str(analisis[0].get_tag())




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
                         True, # NumbersDetection,
                         False, #  PunctuationDetection,
                         True, #  DatesDetection,  --> Setear a True para considerar fechas
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
# Recorro las oraciones (lineas) del archivo
for line in open('ancora_oraciones.txt', encoding="utf-8"):

    # Divido la linea en tokens (palabras)
    palabras_linea = tk.tokenize(line.lower())

    # Agrupo en oraciones para analizar
    for oracion in sp.split(palabras_linea):
        # Analizo la oracion          
        oracion = morfo.analyze(oracion)
        palabras = oracion.get_words()

    # Recorro las palabras de la oracion
        for palabra in palabras:  
            # Actualizo la cantidad de palabras leidas
            i += 1
            # Obtengo el analisis
            analisis = palabra.get_analysis()
            if analisis != ():
                if ("Z" in str(analisis[0].get_tag())):   
                    # la palabra es un numero
                    if es_fecha(palabra.get_form(), tk, sp, morfo):
                        agregar_a_fechas(palabra.get_form())
                    else:
                        agregar_a_numeros(palabra.get_form())
                elif ("W" in str(analisis[0].get_tag())): 
                    # la palabra es una fecha
                    agregar_a_fechas(palabra.get_form())
                else:
                    # no se reconoce la etiqueta de la palabra
                    agregar_a_desconocidos(palabra.get_form(), analisis[0].get_tag())
            else:
                # Agrego palabra al diccionario
                agregar_a_lexicon(palabra.get_form())

  	# # Recorro las palabras de la oracion
   #  for palabra in line[:-3].split(' '):	# el [:-3] es para deshacerme del salto de linea al final de cada oracion
   #    	# Actualizo la cantidad de palabras leidas
   #      i += 1
   #      if palabra.isdigit() and len(palabra) > 0:
   #          agregar_a_numeros(palabra)
   #      else:
   #          agregar_a_lexicon(palabra.lower())
 
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
# resultado = [w for (_,w) in top] 	# Almaceno unicamente las palabras
resultado = [str(f) + ' ' + w for (f,w) in top] # Almaceno palabras y cantidad de usos

# Guardo las palabras en un archivo de texto
open("lexicon_ancora.txt", "w").write(BOM_UTF8 + "\n".join(resultado).encode("utf-8"))


# Palabras identificadas como numero
# - - - - - - - - - - - - - - - - -
numeros_aux = [w for (w,_) in numeros.items()]
open("descarte/lexicon_ancora-numeros.txt", "w").write(BOM_UTF8 + "\n".join(numeros_aux).encode("utf-8"))

# Palabras identificadas como fecha
# - - - - - - - - - - - - - - - - -
fechas_aux = [w for (w,_) in fechas.items()]
open("descarte/es-lexicon_ancora-fechas.txt", "w").write(BOM_UTF8 + "\n".join(fechas_aux).encode("utf-8"))

# Palabras con tag desconocida
# - - - - - - - - - - - - - - 
desconocidos_aux = [str(w) + ' ' + str(t) for (w,t) in desconocidos.items()]
open("descarte/es-lexicon_ancora-desconocidos.txt", "w").write(BOM_UTF8 + "\n".join(desconocidos_aux).encode("utf-8"))


# Calculo porcentaje de cobertura
# -------------------------------
print("La cantidad total de palabras parseadas es: " + str(len(lexicon)))