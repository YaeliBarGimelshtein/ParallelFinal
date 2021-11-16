build:
	mpicxx -fopenmp -c program.c -o program.o
	nvcc -I./inc -c cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -o mpiCudaOpemMP  program.o cudaFunctions.o  /usr/local/cuda/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./mpiCudaOpemMP

run:
	mpiexec -np 3 ./mpiCudaOpemMP <input.txt> output.txt

runOn2:
	mpiexec -np 3 -machinefile  mf  -map-by  node  ./mpiCudaOpemMP


