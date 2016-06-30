from codecs import open, BOM_UTF8
import csv
import sys
import io

sys.stdin = io.TextIOWrapper(sys.stdin.buffer,encoding='utf-8');
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8');

csv.field_size_limit(sys.maxsize)
archivo = open("diez_porciento.csv", "r")
lector = csv.reader(archivo, delimiter=' ')
salida = open("diez_porciento_filtrado.csv", "w");

for r in lector:
	if (len(r) == 13):
		print(str(r), file = salida)