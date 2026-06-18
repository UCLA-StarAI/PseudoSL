#!/bin/bash

for filename in ./grafo/*.txt; do
	FILE="$(basename "$filename" .txt)"
	echo "---"
	echo "using vtree_hg"
	time ./exe/sddSimpath ./grafo/$FILE.txt ./grafo/$FILE.vtree_hg
	echo "---"
	echo "using vtree minfill"
	time ./exe/sddSimpath ./grafo/$FILE.txt ./grafo/$FILE.vtree
	echo "---"
	echo "using graphillion"
	time python ./graphillion/gTut.py < ./grafo/$FILE.txt
done

#for filename in ./grafo/*.txt; do
#    FILE="$(basename "$filename" .txt)"
#    time ./exe/sddSimpath ./grafo/$FILE.txt ./grafo/$FILE.vtree
#done


#for filename in ./grafo/*.txt; do
#    FILE="$(basename "$filename" .txt)"
#    time python ./graphillion/gTut.py < ./grafo/$FILE.txt
#done

