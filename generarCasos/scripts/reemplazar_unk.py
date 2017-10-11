import sys

def separar(linea):
	return linea.replace("\n","").split(",")

original = open(sys.argv[1],"r")
sin_unk = open("Ventana_indizada/" + sys.argv[1],"r")
nuevo = open("Nuevos/" + sys.argv[1],"w")

l_or = original.readline()
l_su = sin_unk.readline()
while l_or != "" and l_su != "":
	if l_or == l_su or "56946" not in l_or:
		nuevo.write(l_or)
	else:
		list_or = separar(l_or)
		list_su = separar(l_su)
		salida = ""
		i = 0
		for n in list_or:
			if n != list_su[i] and n == "56946":
				if i == 0:
					salida = list_su[i]
				else:
					salida += "," + list_su[i]
			else:
				if i == 0:
					salida = n
				else:
					salida += "," + n
			i += 1
		salida += "\n"
		nuevo.write(salida)
	l_or = original.readline()
	l_su = sin_unk.readline()