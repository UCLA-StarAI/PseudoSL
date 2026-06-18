CC = g++ -std=c++11
all: genSdd zddFormat zddFormatGP simpath

genSdd:
	$(CC) -O2 -Wall -I./include   -c -o src/genSdd.o src/genSdd.cpp
	$(CC) -O2 -Wall -I./include src/genSdd.o -L./lib -lsdd -lm -o exe/genSdd

zddFormat:
	$(CC) -O2 -Wall src/zddFormat.cpp -o exe/zddFormat

zddFormatGP:
	$(CC) -O2 -Wall src/zddFormatGP.cpp -o exe/zddFormatGP

simpath:
	$(CC) -O2 -Wall src/simpath.cpp -o exe/simpath

clean:
	rm -f exe/* src/genSdd.o output/*.dot output/*.png output/*.sdd
