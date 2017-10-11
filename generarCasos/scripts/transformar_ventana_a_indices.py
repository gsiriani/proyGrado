import sys
import re

archivo_diccionario = open(sys.argv[1],"r")
archivo_entrada = open(sys.argv[2],"r")
archivo_salida = open(sys.argv[3],"w")
ventana = 11
separador = " "
separador_salida = ","
indice_default = "UNK"
indice_punct = "PUNCT"
indice_num = "NUM"

diccionario = {}
i = 0
for p in archivo_diccionario:
	diccionario[p.replace("\n","")] = str(i)
	i += 1
archivo_diccionario.close()

valor_default = diccionario[indice_default]
valor_punct = diccionario[indice_punct]
valor_num = diccionario[indice_num]

unks = []

for l in archivo_entrada:
	auxL = l.decode('utf8').replace("\n","").encode("utf8")
	separada = auxL.split(separador)
	inicio = separada[:ventana]
	fin = separada[ventana:]
	salida = []
	for p in inicio:
		if re.match("^\W+$",p) and p not in diccionario:
			salida.append(valor_punct)
		elif re.match("^\d+,\d+$",p) and p not in diccionario:
			salida.append(valor_num)
		else:
			salida.append(diccionario.setdefault(p,valor_default))
	salida += fin
	escribir = ""
	for p in salida:
		if escribir == "":
			escribir = p
		else:
			escribir += separador_salida + p
	escribir += "\n"
	archivo_salida.write(escribir)

archivo_entrada.close()
archivo_salida.close()