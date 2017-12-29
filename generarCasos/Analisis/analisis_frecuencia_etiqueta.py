import sys
import os
import re

largo_oracion = 11
separador = ","

def encontrar_uno(etiqueta):
	indice = None
	for i in range(len(etiqueta)):
		if etiqueta[i] == "1":
			indice = i
	return indice

archivo_tags = open(sys.argv[1],"r")
carpeta_entrada = sys.argv[2]
carpeta_salida = sys.argv[3]

if not re.match("./$",carpeta_entrada):
	carpeta_entrada += "/"
if not re.match("./$",carpeta_salida):
	carpeta_salida += "/"

tags = []
for t in archivo_tags:
	tags.append(t.replace("\n",""))

etiquetas = {}

for nombre in os.listdir(carpeta_entrada):
	archivo_entrada = open(carpeta_entrada + nombre, "r")
	archivo_salida = open(carpeta_salida + nombre, "w")
	for linea in archivo_entrada:
		etiqueta = linea.replace("\n","").split(separador)[largo_oracion:]
		indice = encontrar_uno(etiqueta)
		if indice in etiquetas:
			etiquetas[indice] += 1
		else:
			etiquetas[indice] = 1
	for t in etiquetas:
		archivo_salida.write(tags[t] + " : " + str(etiquetas[t]) + "\n")
	for i in range(len(tags)):
		if i not in etiquetas:
			archivo_salida.write(tags[i] + " : 0\n")
	archivo_entrada.close()
	archivo_salida.close()

