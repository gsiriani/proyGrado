import sys
import re
import os

separador = ","
largo_salida = 25

def list_to_str(vector):
	salida = ""
	primero = True
	for p in vector:
		if primero:
			salida = str(p)
			primero = False
		else:
			salida += separador + str(p)
	return salida

c_entrada = sys.argv[1]
c_salida = sys.argv[2]

if not re.match(".*/$",c_entrada):
	c_entrada += "/"
if not re.match(".*/$",c_salida):
	c_salida += "/"

for n_archivo in os.listdir(c_entrada):
	a_entrada = open(c_entrada + n_archivo,"r")
	a_salida = open(c_salida + n_archivo,"w")
	for linea in a_entrada:
		lista = linea.replace("\n","").split(separador)
		largo = len(lista) - largo_salida
		oracion = lista[:largo]
		etiquetas = lista[largo:]
		salida = []
		out = 1
		for i in xrange(0,len(etiquetas) - 1,4):
			if etiquetas[i] == "1" or etiquetas[i + 3] == "1":
				out = 0
				salida.append(1)
			else:
				salida.append(0)
			if etiquetas[i + 1] == "1" or etiquetas[i + 2] == "1":
				out = 0
				salida.append(1)
			else:
				salida.append(0)		
		salida.append(out)
		a_salida.write(list_to_str(oracion) + separador + list_to_str(salida) + "\n")
	a_entrada.close()
	a_salida.close()