CC = g++ -std=c++11
CFLAGS = -Wno-sign-compare
all: sddSimpath

sddSimpath:
	$(CC) $(CFLAGS) -O2 -Wall -I./include   -c -o src/sddSimpath.o src/sddSimpath.cpp
	$(CC) $(CFLAGS) -O2 -Wall -I./include src/sddSimpath.o -L./libLinux -lsdd -lm -o exe/sddSimpath
	$(CC) $(CFLAGS) -O2 src/graphGen.cpp -o exe/graphGen
