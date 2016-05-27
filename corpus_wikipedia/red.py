from vector_palabras import palabras_comunes, generar_vectores_iniciales

p = palabras_comunes("es-lexicon.txt")
vectores = generar_vectores_iniciales(100003, 3)
print p.obtener_indice("wilpharma")
print vectores[p.obtener_indice("wiLpharma")]