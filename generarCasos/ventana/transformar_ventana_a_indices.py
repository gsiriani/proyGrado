import sys
import re
import os

archivo_diccionario = open(sys.argv[1],"r")
ifolder = sys.argv[2]
ofolder = sys.argv[3]
ventana = 11
separador = " "
separador_salida = ","
indice_default = "UNK"
indice_punct = "PUNCT"
indice_num = "NUM"

if not re.match(".*/$",ifolder):
	ifolder += "/"
if not re.match(".*/$",ofolder):
	ofolder += "/"

diccionario = {}
i = 0
for p in archivo_diccionario:
	diccionario[p.replace("\n","")] = str(i)
	i += 1
archivo_diccionario.close()

valor_default = diccionario[indice_default]
valor_punct = diccionario[indice_punct]
valor_num = diccionario[indice_num]

for archivo in os.listdir(ifolder):
	archivo_entrada = open(ifolder + archivo,"r")
	archivo_salida = open(ofolder + archivo,"w")
	for l in archivo_entrada:
		auxL = l.replace("\n","")
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