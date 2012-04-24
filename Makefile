# Assignment: Project 2 CUDA Rasterizer
# Class: CSC 458 & 571
# Instructor: Prof. Lupo, Prof. Wood

CC = nvcc
CFLAGS = 
LD = nvcc
LDFLAGS = -lm -lpng
OBJECTS = main.o linear_math.o mesh_loader.o png_loader.o rasterizer.o structures.o

all: rasterizer

rasterizer: $(OBJECTS)
	$(LD) $(LDFLAGS) $(OBJECTS) -o rasterizer

%.o: %.cu
	$(CC) $(CCFLAGS) -c -arch=sm_11 $<

cpu: all
	./rasterizer

gpu: all
	./rasterizer -cuda

clean:
	rm -f *.o
	rm -f rasterizer
	rm -f test.png
