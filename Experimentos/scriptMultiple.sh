cd Experimento28
cd Ner
cd lr_0.1
pwd 
python ner.py > salida.txt
cd ..
cd lr_0.07
pwd
python ner.py > salida.txt
cd ..
cd ..
cd ..
cd Experimento29
cd EmbeddingAleatorio
cd lr_0.1
pwd
python chunking.py > salida.txt
cd ..
cd lr_0.01
pwd
python chunking.py > salida.txt
cd ..
cd lr_0.03
pwd
python chunking.py > salida.txt
cd ..
cd lr_0.1_150features
pwd
python chunking.py > salida.txt
cd ..
echo 'FIN!!!!'