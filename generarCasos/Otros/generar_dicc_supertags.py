import sys
import re

archivo = open(sys.argv[1],"r")
salida = open(sys.argv[2],"w")
supertags = []
for linea in archivo:
	if linea != "\n":
		supertag = linea.replace("\n","")
		supertag = re.sub("^.*\t","",supertag)
		if supertag not in supertags and supertag != "":
			salida.write(supertag + "\n")
			supertags.append(supertag)
archivo.close()
salida.close()