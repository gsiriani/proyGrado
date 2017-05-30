## ----------------------------------------------
## -------------    MAIN PROGRAM  ---------------
## ----------------------------------------------

import freeling
import sys
import random
import io
import os
from codecs import open, BOM_UTF8

def generar_caso_w(indice, s, erronea, cant):
  window = "";
  window_mal = "";
  for j in range(cant - indice):
    window = window + "OUT ";
    window_mal = window_mal + "OUT ";
  for j in range(max(indice - cant, 0), indice):
    window = window + s[j].lower() + " ";
    window_mal = window_mal + s[j].lower() + " ";
  for j in range(indice,min(len(s), cant + indice + 1)):
    window = window + s[j].lower() + " ";
    if (j == indice):
      window_mal = window_mal + erronea + " ";
    else:
      window_mal = window_mal + s[j].lower() + " ";
  for j in range(cant - len(s) + indice + 1):
    window = window + "OUT "
    window_mal = window_mal + "OUT "
  window = window + "1\n"
  window_mal = window_mal + "0\n"
  return [window, window_mal]

def generar_caso_s(indice, s, erronea):
  oracion = str(len(s)) + " "
  oracion_mal = str(len(s)) + " "
  for j in range(len(s)):
    oracion += s[j].lower() + " "
    if j == indice:
      oracion_mal += erronea + " "
    else:
      oracion_mal += s[j].lower() + " "
  for k in range(len(s)):
    oracion += str(k - indice) + " "
    oracion_mal += str(k - indice) + " "
  oracion = oracion + "1\n"
  oracion_mal = oracion_mal + "0\n"
  return [oracion, oracion_mal]



def procesar_lm(s, p_comunes, cant):
  output = [[],[]]
  oracion = []
  oracion_aux = s.split(" ")
  for p in oracion_aux:
    if p != "":
      oracion.append(p)
  for i in range(len(oracion)):
    p_mal = random.randint(0, len(p_comunes) - 1)
    window = generar_caso_w(i, oracion, p_comunes[p_mal], cant)
    sentece = generar_caso_s(i, oracion, p_comunes[p_mal])
    output[0] += window
    output[1] += sentece
  return output


## Modify this line to be your FreeLing installation directory

p_comunes = [];
for p in open("lexicon_total.txt", encoding="latin-1"):
      p_comunes.append(p.replace("\n",""));

FREELINGDIR = "/usr/local";

#sys.stdin = io.TextIOWrapper(sys.stdin.buffer,encoding='latin-1');
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8');

DATA = FREELINGDIR+"/share/freeling/";
LANG="es";

freeling.util_init_locale("default");

# create language analyzer
la=freeling.lang_ident(DATA+"common/lang_ident/ident.dat");

# create options set for maco analyzer. Default values are Ok, except for data files.
op= freeling.maco_options("es");
op.set_data_files( "", 
                   DATA + "common/punct.dat",
                   DATA + LANG + "/dicc.src",
                   DATA + LANG + "/afixos.dat",
                   "",
                   DATA + LANG + "/locucions.dat", 
                   DATA + LANG + "/np.dat",
                   DATA + LANG + "/quantities.dat",
                   DATA + LANG + "/probabilitats.dat");

# create analyzers
tk=freeling.tokenizer(DATA+LANG+"/tokenizer.dat");
sp=freeling.splitter(DATA+LANG+"/splitter.dat");
sid=sp.open_session();


train_w = open("/media/gonzalo/083E1C113E1BF702/Users/Gonzalo/Documents/Resultados/lm_training_w.csv","w",encoding='latin-1')
test_w = open("/media/gonzalo/083E1C113E1BF702/Users/Gonzalo/Documents/Resultados/lm_testing_w.csv","w",encoding='latin-1')
pruebas_w = open("/media/gonzalo/083E1C113E1BF702/Users/Gonzalo/Documents/Resultados/lm_pruebas_w.csv","w",encoding='latin-1')

train_s = open("/media/gonzalo/083E1C113E1BF702/Users/Gonzalo/Documents/Resultados/lm_training_s.csv","w",encoding='latin-1')
test_s = open("/media/gonzalo/083E1C113E1BF702/Users/Gonzalo/Documents/Resultados/lm_testing_s.csv","w",encoding='latin-1')
pruebas_s = open("/media/gonzalo/083E1C113E1BF702/Users/Gonzalo/Documents/Resultados/lm_pruebas_s.csv","w",encoding='latin-1')

folder = sys.argv[1]

for archivo in os.listdir(folder):
  # process input text
  print ("\n" + archivo)
  entrada = open(folder + "/" + archivo, "r", encoding='latin-1')

  line = entrada.readline();

  linea = "";
  sentences = []
  print ('Arranca')
  sys.stdout.flush();
  i = 0;
  corte = False

  while (line) :
    if (line == "\n" or line.startswith(("<doc", "</doc>", "ENDOFARTICLE", "REDIRECT", "Acontecimientos", "Fallecimientos", "Nacimientos", " Acontecimientos", " Fallecimientos", " Nacimientos"))):
      line = entrada.readline();
      corte = True;
      continue;
    else:
      if corte:
        i = i + 1 
        corte = False;
        l = tk.tokenize(linea);
        ls = [];
        ls = sp.split(sid,l,False);
        for s in ls:
          oracion = ""
          for p in range(len(s)):
            oracion = oracion + s[p].get_form() + " ";
          sentences.append(oracion)
        linea = ""
      linea = linea + " " + line;
      line=entrada.readline();


    
        
  sys.stdout.flush()
  l = tk.tokenize(linea);

  largo = 11;
  ls = []
  ls = sp.split(sid,l,False);
  for s in ls:
    oracion = ""
    for p in range(len(s)):
      oracion = oracion + s[p].get_form() + " ";
    sentences.append(oracion)
  cant = int((largo - 1) / 2);
  print ('Crear oraciones')
  sys.stdout.flush()

  for s in sentences:
    r = random.random()
    output = procesar_lm(s, p_comunes, cant)
    if r <= 0.7:
      for o in output[0]:
        train_w.write(o)
      for o in output[1]:
        train_s.write(o)
    elif r <= 0.85:
      for o in output[0]:
        pruebas_w.write(o)
      for o in output[1]:
        pruebas_s.write(o)
    else:
      for o in output[0]:
        test_w.write(o)
      for o in output[1]:
        test_s.write(o)
  entrada.close()
sp.close_session(sid);
