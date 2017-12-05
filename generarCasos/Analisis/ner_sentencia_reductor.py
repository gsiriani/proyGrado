import sys
import os

separador = ","
cantidad_previos = 1
largo_salida = 17

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
	largo = len(lista) - largo_salida
	etiquetas = lista[largo:]
	casos_oracion.append((linea,etiquetas))
	if largo == (int(lista[0]) + cantidad_previos + 1):
		if any(map(lambda x: not es_out(x[1]),casos_oracion)):
			salida += map(lambda x: x[0],casos_oracion)
		casos_oracion = []

for caso in salida:
	a_salida.write(caso)

a_entrada.close()
a_salida.close()