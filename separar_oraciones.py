## ----------------------------------------------
## -------------    MAIN PROGRAM  ---------------
## ----------------------------------------------

import freeling
import sys
import random
import io
from codecs import open, BOM_UTF8

## Modify this line to be your FreeLing installation directory

p_comunes = [];
for p in open("es-lexicon.txt", encoding="latin-1"):
      p_comunes.append(p);

FREELINGDIR = "/usr/local";

sys.stdin = io.TextIOWrapper(sys.stdin.buffer,encoding='latin-1');
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8');

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

# process input text
line=sys.stdin.readline();

linea = "";
sentences = []
sys.stderr.write('Arranca\n');
sys.stderr.flush();
i = 0;
corte = False
while (line) :
  if (line == "\n" or line.startswith(("<doc", "</doc>", "ENDOFARTICLE", "REDIRECT","Acontecimientos", "Fallecimientos", "Nacimientos"))):
    line=sys.stdin.readline();
    corte = True;
    continue;
  if corte:
    corte = False;
    l = tk.tokenize(linea);
    ls = sp.split(sid,l,False);
    sentences = sentences + ls;
    linea = ""
  linea = linea + " " + line;
  line=sys.stdin.readline();
      
sys.stderr.write('Tokenizar\n'); 
sys.stderr.flush();
l = tk.tokenize(linea);

largo = 11;
ls = sp.split(sid,l,False);
sentences = sentences + ls;
cant = int((largo - 1) / 2);
sys.stderr.write('Crear oraciones');
sys.stderr.flush();
for s in sentences:
  oracion = ""
  for p in range(len(s))
    oracion = oracion + s[p].get_form().lower() + " ";
  print oracion
#  for i in range(len(s)):
#    p_mal = random.randint(0, 99999);
#    window = "";
#    window_mal = "";
#    for j in range(cant - i):
#      window = window + "OUT ";
#      window_mal = window_mal + "OUT ";
#    for j in range(max(i - cant, 0), i):
#      window = window + s[j].get_form().lower() + " ";
#      window_mal = window_mal + s[j].get_form().lower() + " ";
#    for j in range(i,min(len(s), cant + i + 1)):
#      window = window + s[j].get_form().lower() + " ";
#      if (j == i):
#        window_mal = window_mal + p_comunes[p_mal] + " ";
#      else:
#        window_mal = window_mal + s[j].get_form().lower() + " ";
#    for j in range(cant - len(s) + i + 1):
#      window = window + "OUT "
#      window_mal = window_mal + "OUT "
#    window = window + "1 0"
#    window_mal = window_mal + "0 1"
#    print (window);
#    print (window_mal);
sp.close_session(sid);
    
