cd MacroChunking
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py macrochunking 3 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py macrochunking 3 p > salida.txt
cd ..
cd ..
cd MicroChunking
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py microchunking 3 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py microchunking 3 p > salida.txt
cd ..
cd ..
cd Ner
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py ner 3 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py ner 3 p > salida.txt
cd ..
cd ..
cd Pos
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py pos 3 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py pos 3 p > salida.txt
cd ..
cd ..
cd Srl
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva_srl.py srl 3 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py srl 3 p > salida.txt
cd ..
cd ..
cd St_Reducido
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py supertag_reducido 3 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py supertag_reducido 3 p > salida.txt
cd ..
cd ..
cd St_Completo
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py supertag_completo 3 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py supertag_completo 3 p > salida.txt
cd ..
cd ..
cd Combinadas
cd St_Reducido
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../../multired.py r 3 a > salida.txt
cd ..
cd Precalculado
pwd
date "+%H:%M:%S   %d/%m/%y" 
python ../../../multired.py r 3 p > salida.txt
cd ..
cd ..
cd St_Completo
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../../multired.py a 3 a > salida.txt
cd ..
cd Precalculado
pwd
date "+%H:%M:%S   %d/%m/%y" 
python ../../../multired.py a 3 p > salida.txt
echo 'FIN!!!'
