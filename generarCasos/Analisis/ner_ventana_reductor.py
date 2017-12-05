import sys
import re
import os

separador = ","
largo_ventana = 11
out = "56945"
out_ind = out + "," + out + "," + out + "," + out + "," + out

def es_out(vector):
	out = True
	for i in range(len(vector) - 1):
		if vector[i] != "0":
			out = False
	return out

a_entrada = open(sys.argv[1],"r")
a_salida = open(sys.argv[2],"w")

casos_oracion = []
salida = []

for linea in a_entrada:
	lista = linea.replace("\n","").split(separador)
	etiquetas = lista[largo_ventana:]
	casos_oracion.append((linea,etiquetas))
	if re.match("^.+," + out_ind + ",.*",linea):
		if any(map(lambda x: not es_out(x[1]),casos_oracion)):
			salida += map(lambda x: x[0],casos_oracion)
		casos_oracion = []

for caso in salida:
	a_salida.write(caso)

a_entrada.close()
a_salida.close()