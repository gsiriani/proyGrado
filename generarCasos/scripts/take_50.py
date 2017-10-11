import sys

entrada = sys.argv[1]
salida = sys.argv[2]
f_entrada = open(entrada,"r")
f_salida = open(salida,"w")

i = 0
for l in f_entrada:
	f_salida.write(l)
	i += 1
	if i >= 50:
		break