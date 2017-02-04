

from codecs import open, BOM_UTF8

# Funciones auxiliares
# --------------------
def has_num(palabra):
  return any(char.isdigit() for char in palabra)

# Estructura principal
# --------------------
oraciones = ''

# Recorro las oraciones (lineas) del archivo
for line in open('ancora_oraciones.txt', encoding="utf-8"):
    linea = 'OUT '

    # Recorro las palabras de la oracion
    for token in line[:-3].split(' '):	# el [:-3] es para deshacerme del salto de linea al final de cada oracion

        # Obtengo palabras separadas por '_'
        palabras = token.split('_')

        for palabra in palabras:

            if has_num(palabra):
                linea += 'NUM '

            elif len(palabra) > 0:
                linea += palabra.lower()
