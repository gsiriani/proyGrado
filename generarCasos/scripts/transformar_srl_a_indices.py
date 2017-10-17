import sys
import re
import os

archivo_diccionario = open(sys.argv[1],"r")
ifolder = sys.argv[2]
ofolder = sys.argv[3]
separador = " "
separador_salida = ","
indice_default = "UNK"
indice_punct = "PUNCT"
indice_num = "NUM"
indice_date = "DATE"

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
valor_date = diccionario[indice_date]

for archivo in os.listdir(ifolder):
  archivo_entrada = open(ifolder + archivo,"r")
  archivo_salida = open(ofolder + archivo,"w")
  for l in archivo_entrada:
    auxL = l.replace("\n","")
    separada = auxL.split(separador)
    oracion = eval(separada[0])
    inicio = map(lambda x: x[0],oracion)
    fin = separada[1:]
    salida = []
    for p in inicio:
      if (re.match("^\W+$",p) and p not in diccionario) or p == indice_punct:
        salida.append(valor_punct)
      elif p == indice_date:
        salida.append(valor_date)
      elif p == indice_default:
        salida.append(valor_default)
      elif (re.match("^\d+,\d+$",p) and p not in diccionario) or p == indice_num:
        salida.append(valor_num)
      else:
        salida.append(diccionario.setdefault(p.decode("utf-8").lower().encode("utf-8"),valor_default))
    salida += fin
    escribir = str(oracion[0][1]) + separador_salida + str(oracion[0][2])
    for p in salida:
      escribir += separador_salida + p
    escribir += "\n"
    archivo_salida.write(escribir)
  archivo_entrada.close()
  archivo_salida.close()