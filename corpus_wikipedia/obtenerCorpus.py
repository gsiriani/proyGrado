import freeling
import sys
import random
import io
from codecs import open, BOM_UTF8
from glob import glob

def palabras_frecuentes(words_in_corpus = 130000000, max_words = 100000, start=0):
    print "Ejecutando..."
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


    FREELINGDIR = "/usr/local";

    sys.stdin = io.TextIOWrapper(sys.stdin.buffer,encoding='latin-1');
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='latin-1');

    DATA = FREELINGDIR+"/share/freeling/";
    LANG="es";

    freeling.util_init_locale("default");

    # create language analyzer
    la=freeling.lang_ident(DATA+"common/lang_ident/ident.dat");

    # create options set for maco analyzer. Default values are Ok, except for data files.
    op= freeling.maco_options("es");
    op.set_data_files( "", 
                       DATA + "common/punct.dat",
                       DATA + LANG + "/dicc.src",
                       DATA + LANG + "/afixos.dat",
                       "",
                       DATA + LANG + "/locucions.dat", 
                       DATA + LANG + "/np.dat",
                       DATA + LANG + "/quantities.dat",
                       DATA + LANG + "/probabilitats.dat");

    # create analyzers
    tk=freeling.tokenizer(DATA+LANG+"/tokenizer.dat");

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
            palabras_linea = map(lambda x: x.getform(),tk.tokenize(line.lower()))

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
     
    # Ordeno segun la frecuencia y me quedo con la cantidad deseada        
    top = []  
    for (w,freq) in lexicon.items():    
        top.append((freq, w))
     
    top = sorted(top, reverse=True)[:max_words] # top max_words
    resultado = [w for (_,w) in top] 

    # Guardo las palabras en un archivo de texto
    open("es-lexicon.txt", "w").write(BOM_UTF8 + "\n".join(resultado).encode("utf-8"))


    # Calculo porcentaje de cobertura
    porcentaje = (100*max_words)/len(lexicon)
    print "El porcentaje de cobertura es: " + str(porcentaje)
    print motivo_fin


def obtener_ventanas(file_name, window_size):
    return []



palabras_frecuentes()