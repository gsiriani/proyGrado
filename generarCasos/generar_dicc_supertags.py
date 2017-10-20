import sys
import re

archivo = open(sys.argv[1],"r")
salida = open(sys.argv[2],"w")
supertags = {}
for linea in archvio:
	if linea != "\n":
		supertag = linea.replace("\n","")
		supertag = re.sub("^.*\t","",supertag)
		salida.write(supertag + "\n")
archivo.close()
salida.close()