import sys

archivo = open(sys.argv[1],"r")
salida = open(sys.argv[2],"w")

def a_csv(lista):
	retorno = ""
	for elemento in lista:
		if retorno == "":
			retorno = elemento
		else:
			retorno += "," + elemento
	return retorno

for linea in archivo:
	separado = linea.replace("\n","").replace("\r","").split(",")
	if any(map(lambda x: x == "1", separado[11:])):
		separado.append("0")
	else:
		separado.append("1")
	l_salida = a_csv(separado)
	salida.write(l_salida + "\n")

archivo.close()
salida.close()