from glob import glob
from codecs import open, BOM_UTF8   

def palabras_frecuentes(words_in_corpus = 60000000, max_words = 100000, start=0):
    lexicon = {}
    i = 0
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
                break

            # Divido la linea en palabras
            palabras_linea = line.lower().split()

            # Actualizo la cantidad de palabras leidas
            i += len(palabras_linea)

            # Agrego las palabras al diccionario
            for palabra in palabras_linea:
                # conservo unicamente caracteres alfanumericos
                palabra = ''.join(e for e in palabra if e.isalnum())
                # ignoro numeros y palabras de largo 0
                if palabra.isdigit() or len(palabra) == 0:
                    continue
                if lexicon.has_key(palabra):
                    lexicon[palabra] += 1
                else:
                    lexicon[palabra] = 1
     
    # Ordeno segun la frecuencia y me quedo con la cantidad deseada        
    top = []  
    for (w,freq) in lexicon.items():    
        top.append((freq, w))
     
    top = sorted(top, reverse=True)[:max_words] # top max_words
    resultado = [w for (_,w) in top] 

    # Guardo las palabras en un archivo de texto
    open("es-lexicon.txt", "w").write(BOM_UTF8 + "\n".join(resultado).encode("utf-8"))


def obtener_ventanas(file_name, window_size):
    return []
