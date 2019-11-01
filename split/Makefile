all: vecadd

vecadd: vecadd.cu
	nvcc -std=c++11 -o $@ $< -lnvToolsExt

clean:
	rm -f *.o vecadd
