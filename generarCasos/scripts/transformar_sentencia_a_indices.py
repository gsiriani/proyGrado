import sys
import re

entrada = open(sys.argv[1],"r")
f_salida = open(sys.argv[2],"w")
f_diccionario = open(sys.argv[3],"r")

indice_default = "UNK"
indice_punct = "PUNCT"
indice_num = "NUM"

diccionario = {}
i = 0
for p in f_diccionario:
  diccionario[p.replace("\n","")] = str(i)
  i += 1
f_diccionario.close()

valor_default = diccionario[indice_default]
valor_punct = diccionario[indice_punct]
valor_num = diccionario[indice_num]

for line in entrada:
  separado = line.replace("\n","").split(" ")
  lista_palabras = eval(separado[0])
  oracion = []
  for e in lista_palabras:
    oracion.append(e[0].decode('utf-8').lower().encode('utf-8'))
  salida = [str(lista_palabras[0][1])]
  for p in oracion:
    if re.match("^\W+$",p) and p not in diccionario:
      salida.append(valor_punct)
    elif re.match("^\d+,\d+$",p) and p not in diccionario:
      salida.append(valor_num)
    else:
      salida.append(diccionario.setdefault(p,valor_default))
  salida += separado[1:]
  string_salida = ""
  for p in salida:
    if string_salida == "":
      string_salida = p
    else:
      string_salida += "," + p
  string_salida += "\n"
  f_salida.write(string_salida)
f_salida.close()
entrada.close()