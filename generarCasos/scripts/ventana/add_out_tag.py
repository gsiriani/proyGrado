import sys
import os
import re

c_entrada = sys.argv[1]
c_salida = sys.argv[2]

def a_csv(lista):
	retorno = ""
	for elemento in lista:
		if retorno == "":
			retorno = elemento
		else:
			retorno += "," + elemento
	return retorno

if not re.match("^.*/$",c_entrada):
	c_entrada += "/"
if not re.match("^.*/$",c_salida):
	c_salida += "/"

for n_archivo in os.listdir(c_entrada):
	a_entrada = open(c_entrada + n_archivo,"r")
	a_salida = open(c_salida + n_archivo,"w")
	for linea in a_entrada:
		separado = linea.replace("\n","").replace("\r","").split(",")
		if any(map(lambda x: x == "1", separado[11:])):
			separado.append("0")
		else:
			separado.append("1")
		l_salida = a_csv(separado)
		salida.write(l_salida + "\n")

	a_entrada.close()
	a_salida.close()