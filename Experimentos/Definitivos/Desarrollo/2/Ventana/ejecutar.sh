cd MacroChunking
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
# python ../../ventana.py macrochunking 20 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
# python ../../ventana.py macrochunking 20 p > salida.txt
cd ..
cd ..
cd MicroChunking
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
# python ../../ventana.py microchunking 20 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
# python ../../ventana.py microchunking 20 p > salida.txt
cd ..
cd ..
cd Ner
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
# python ../../ventana.py ner 20 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
# python ../../ventana.py ner 20 p > salida.txt
cd ..
cd ..
cd Pos
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
# python ../../ventana.py pos 20 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
# python ../../ventana.py pos 20 p > salida.txt
cd ..
cd ..
cd St_Reducido
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
# python ../../ventana.py supertag_reducido 20 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
# python ../../ventana.py supertag_reducido 20 p > salida.txt
cd ..
cd ..
cd St_Completo
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
# python ../../ventana.py supertag_all 20 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
# python ../../ventana.py supertag_all 20 p > salida.txt
cd ..
cd ..
cd Combinadas
cd St_Reducido
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../../multired.py r 20 a > salida.txt
cd ..
cd Precalculado
pwd
date "+%H:%M:%S   %d/%m/%y" 
python ../../../multired.py r 20 p > salida.txt
cd ..
cd ..
cd St_Completo
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../../multired.py a 20 a > salida.txt
cd ..
cd Precalculado
pwd
date "+%H:%M:%S   %d/%m/%y" 
python ../../../multired.py a 20 p > salida.txt
echo 'FIN!!!'
