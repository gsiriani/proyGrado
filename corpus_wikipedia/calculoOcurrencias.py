from codecs import open, BOM_UTF8
import csv
import sys
import io
import re


csv.field_size_limit(sys.maxsize)
archivo = open("descarte/es-lexicon-total.txt", 'r', encoding="utf-8")																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				

i = 0
ocurrencias = 0
cant_ocurrencias = []
for l in archivo:
	r = l.split()
	num = int(re.sub("[^0-9]", "", r[0]))
	ocurrencias = ocurrencias + num
	i = i + 1
	if (i % 5000 == 0):
		cant_ocurrencias.append((i,ocurrencias))

print '#Palabras \t%Palabras \t#Ocurrencias \t%Ocurrencias'
for (n,c) in cant_ocurrencias:
	pp = float(n)*100/i
	po = float(c)*100/ocurrencias
	if n < 1000000:
		print "%u \t\t%f \t%u \t%f" % (n,pp,c,po) 
	else:		
		print "%u \t%f \t%u \t%f" % (n,pp,c,po) 
print '\nTotal de ocurrencias: ' + str(ocurrencias)
print 'Total de palabras: ' + str(i)
