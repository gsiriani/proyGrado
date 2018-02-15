import sys
import re

ventana = 11
out_tag = "OUT"
separador = ","

lexicon_file = open(sys.argv[1], "r")
entrada = open(sys.argv[2],"r")

out_ind = None
i = 0
for line in lexicon_file:
	if line.replace("\n","") == out_tag:
		out_ind = str(i)
		break
	i += 1

lexicon_file.close()

regex = "^"
for i in range(ventana/2):
	regex += out_ind + ","

cantidad = 0
for line in entrada:
	if re.match(regex,line):
		cantidad += 1

print cantidad