import sys
import os
import re

ventana = 11
separador = ","

c_entrada = sys.argv[1]
a_salida = open(sys.argv[2],"w")

if not re.match("^.*/",c_entrada):
	c_entrada += "/"

lista_palabras = []

for n_archivo in os.listdir(c_entrada):
	archivo = open(c_entrada + n_archivo,"r")
	for line in archivo:
		palabras = line.replace("\n","").split(separador)[:ventana]
		for p in palabras:
			if p not in lista_palabras:
				lista_palabras.append(p)
	archivo.close()

for p in lista_palabras:
	a_salida.write(p + "\n")

a_salida.close()