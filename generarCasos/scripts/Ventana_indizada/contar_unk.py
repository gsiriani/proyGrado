import sys

archivo = open(sys.argv[1],"r")
archivo2 = open("Ventana_indizada/" + sys.argv[1], "r")
archivo3 = open("Nuevos/" + sys.argv[1], "r")
i = 0
for l in archivo:
	if "56946" in l:
		i += 1
print i
i = 0
for l in archivo2:
	if "56946" in l:
		i += 1
print i
i = 0
for l in archivo3:
	if "56946" in l:
		i += 1
print i