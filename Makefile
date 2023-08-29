CC	= nvcc
CFLAGS = -I/home/pilsungk/cuda-samples/Common 
DEPS = prob_params.h
OBJ = ssa.o ssa_kernel.o

all: ssa

ssa: $(OBJ)	
	$(CC) -o $@ $^

%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: clean

clean:
	rm -f ssa *.o
