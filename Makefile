.PHONY: default clean

CC = clang

CFLAGS = \
	-std=c2x \
	-O2 \
	-march=native \
	-Wall \
	-Wextra \
	-Wconversion \
	-Wdouble-promotion \
	-Wno-fixed-enum-extension \
	-Wno-shift-op-parentheses \
	-ffp-contract=off \
	-fno-math-errno \
	-fno-slp-vectorize \
	-fno-omit-frame-pointer

#	-fsanitize=undefined \

default: pcr

pcr: main.c render.h render.o prospero.c
	$(CC) -o $@ $< render.o $(CFLAGS) -lomp -L/opt/homebrew/opt/libomp/lib

render.o: render.c render.h simd.h
	$(CC) -c -o $@ $< $(CFLAGS) -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include

render.s: render.c render.h simd.h
	$(CC) -S -o $@ $< $(CFLAGS) -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include

prospero.c: prospero.vm preprocess.py
	./preprocess.py < $< > $@

prospero.vm:
	curl 'https://raw.githubusercontent.com/mkeeter/fidget/refs/heads/main/models/prospero.vm' > $@

clean:
	rm -f pcr
	rm -f render.o
	rm -f render.s
	rm -f prospero.c
