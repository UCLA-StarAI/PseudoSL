#!/bin/bash

time(
for filename in ./vtrees/graphs/*.txt; do
	FILE="$(basename "$filename" .txt)"
	./exe/sddSimpath ./vtrees/graphs/$FILE.txt ./vtrees/vtrees-hg/$FILE.cnf.vtree
done
)


time(
for filename in ./vtrees/graphs/*.txt; do
    FILE="$(basename "$filename" .txt)"
    python ./graphillion/gTut.py < ./vtrees/graphs/$FILE.txt
done
)

