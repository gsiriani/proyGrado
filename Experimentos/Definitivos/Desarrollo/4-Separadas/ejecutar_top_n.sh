cd Ventana
cd Supertag
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
#python ../../top_n.py supertag a > salida_top_n.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
#python ../../top_n.py supertag p > salida_top_n.txt
cd ..
cd ..
cd Supertag_Compacto
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
#python ../../top_n.py supertag_compacto a > salida_top_n.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
#python ../../top_n.py supertag_compacto p > salida_top_n.txt
cd ..
cd ..

cd Combinadas
cd Supertag
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
#python ../../../multired_top_n.py 1 a > salida_top_n.txt
cd ..
cd Precalculado
pwd
date "+%H:%M:%S   %d/%m/%y" 
#python ../../../multired_top_n.py 1 p > salida_top_n.txt
cd ..
cd ..
cd Supertag_Compacto
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
#python ../../../multired_top_n.py 0 a > salida_top_n.txt
cd ..
cd Precalculado
pwd
date "+%H:%M:%S   %d/%m/%y" 
#python ../../../multired_top_n.py 0 p > salida_top_n.txt
cd ..
cd ..
cd ..
cd ..
echo 'FIN VENTANA!!!'


cd Sentencia
cd Supertag
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../top_n.py supertag a > salida_top_n.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../top_n.py supertag p > salida_top_n.txt
cd ..
cd ..
cd Supertag_Compacto
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
python ../../top_n.py supertag_compacto a > salida_top_n.txt
cd ..
cd Precalculado
pwd 
date "+%H:%M:%S   %d/%m/%y"
python ../../top_n.py supertag_compacto p > salida_top_n.txt
cd ..
cd ..

cd Combinadas
cd Supertag
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
#python ../../../multired_top_n.py 1 a > salida_top_n.txt
cd ..
cd Precalculado
pwd
date "+%H:%M:%S   %d/%m/%y" 
#python ../../../multired_top_n.py 1 p > salida_top_n.txt
cd ..
cd ..
cd Supertag_Compacto
cd Aleatorio
pwd
date "+%H:%M:%S   %d/%m/%y"
#python ../../../multired_top_n.py 0 a > salida_top_n.txt
cd ..
cd Precalculado
pwd
date "+%H:%M:%S   %d/%m/%y" 
#python ../../../multired_top_n.py 0 p > salida_top_n.txt
cd ..
cd ..

echo 'FIN!!!'
