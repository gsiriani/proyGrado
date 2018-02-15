cd 3
cd Sentencia
cd Combinadas
cd St_Completo_Compactado
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../../multired.py 2 10 a > salida.txt
cd ..
cd Precalculado
pwd
date "+%H:%M:%S   %d/%m/%y" 
python ../../../multired.py 2 10 p > salida.txt
cd ../../../../..
cd 4-Separadas


cd Ventana
cd MacroChunking
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../ventana.py macrochunking 20 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../ventana.py macrochunking 20 p > salida.txt
cd ..
cd ..
cd MicroChunking
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../ventana.py microchunking 20 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../ventana.py microchunking 20 p > salida.txt
cd ..
cd ..
cd Ner
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../ventana.py ner 20 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../ventana.py ner 20 p > salida.txt
cd ..
cd ..
cd Pos
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../ventana.py pos 20 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../ventana.py pos 20 p > salida.txt
cd ..
cd ..
cd Supertag
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
#python ../../ventana.py supertag 20 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
#python ../../ventana.py supertag 20 p > salida.txt
cd ..
cd ..
cd Supertag_Compacto
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../ventana.py supertag_compacto 20 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../ventana.py supertag_compacto 20 p > salida.txt
cd ..
cd ..
cd Combinadas
cd Supertag
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
#python ../../../multired.py 0 10 a > salida.txt
cd ..
cd Precalculado
pwd
date "+%H:%M:%S   %d/%m/%y" 
#python ../../../multired.py 0 10 p > salida.txt
cd ..
cd ..
cd Supertag_Compacto
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../../multired.py 1 10 a > salida.txt
cd ..
cd Precalculado
pwd
date "+%H:%M:%S   %d/%m/%y" 
python ../../../multired.py 1 10 p > salida.txt
cd ..
cd ..
cd ..
cd ..
echo 'FIN VENTANA!!!'


cd Sentencia
cd MacroChunking
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py macrochunking 20 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py macrochunking 20 p > salida.txt
cd ..
cd ..
cd MicroChunking
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py microchunking 20 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py microchunking 20 p > salida.txt
cd ..
cd ..
cd Ner
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py ner 20 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py ner 20 p > salida.txt
cd ..
cd ..
cd Pos
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py pos 20 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py pos 20 p > salida.txt
cd ..
cd ..
cd Srl
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva_srl.py srl 20 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva_srl.py srl 20 p > salida.txt
cd ..
cd ..
cd Supertag
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
#python ../../convolutiva.py supertag 20 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
#python ../../convolutiva.py supertag 20 p > salida.txt
cd ..
cd ..
cd Supertag_Compacto
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py supertag_compacto 20 a > salida.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../convolutiva.py supertag_compacto 20 p > salida.txt
cd ..
cd ..
cd Combinadas
cd Supertag
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
#python ../../../multired.py 0 10 a > salida.txt
cd ..
cd Precalculado
pwd
date "+%H:%M:%S   %d/%m/%y" 
#python ../../../multired.py 0 10 p > salida.txt
cd ..
cd ..
cd Supertag_Compacto
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../../multired.py 1 10 a > salida.txt
cd ..
cd Precalculado
pwd
date "+%H:%M:%S   %d/%m/%y" 
python ../../../multired.py 1 10 p > salida.txt
cd ..
cd ..
echo 'FIN!!!'
