cd Experimento24
echo 'Experimento 24 - chunking-iobes separado embedding aleatorio'
python chunking.py > ./salida.txt
cd ..
cd Experimento25
echo 'Experimento 25 - ner-iobes separado embedding aleatorio'
python ner.py > ./salida.txt
cd ..
echo 'Fin!!!!!!!!'