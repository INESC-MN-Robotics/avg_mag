all: kernel_v2_61.o cu_veclib.o
	nvcc -arch=sm_61 kernel_v2_61.o cu_veclib.o -o magcomp_v2

kernel_v2_61.o : kernel_v2_61.cu
	nvcc -arch=sm_61 -dc kernel_v2_61.cu

cu_veclib.o : cu_veclib.cu
	nvcc -arch=sm_61 -dc cu_veclib.cu

clean :
	rm -rf kernel_v2_61.o cu_veclib.o magcomp_v2
