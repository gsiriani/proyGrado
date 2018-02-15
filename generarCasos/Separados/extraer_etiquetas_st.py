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
archivo_salida = open(sys.argv[3],"w")

if not re.match("./$",carpeta_entrada):
	carpeta_entrada += "/"

tags = []
for t in archivo_tags:
	tags.append(t.replace("\n",""))

etiquetas = {}

for nombre in os.listdir(carpeta_entrada):
	archivo_entrada = open(carpeta_entrada + nombre, "r")
	for linea in archivo_entrada:
		etiqueta = linea.replace("\n","").split(separador)[largo_oracion:]
		indice = encontrar_uno(etiqueta)
		if indice in etiquetas:
			etiquetas[indice] += 1
		else:
			etiquetas[indice] = 1
	archivo_entrada.close()
for t in etiquetas:
	if etiquetas[t] >= 30 and re.match("^v.+$",tags[t]):
		archivo_salida.write(tags[t] + "\n")
	elif etiquetas[t] >= 20 and re.match("^n.+$",tags[t]):
		archivo_salida.write(tags[t] + "\n")
	elif etiquetas[t] >= 10 and re.match("^a.+$",tags[t]):
		archivo_salida.write(tags[t] + "\n")
	elif not re.match("^a.+$",tags[t]) and not re.match("^a.+$",tags[t]) and not re.match("^v.+$",tags[t]):
		archivo_salida.write(tags[t] + "\n")
archivo_salida.write("v-unknown\nn-unknown\na-unknown\n")
archivo_salida.close()