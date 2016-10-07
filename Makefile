
compile:
	nvcc -Wno-deprecated-gpu-targets -o radix ./radix.cu

clean:
	rm -f ./radix

radix:
	make -s recompile
	./radix
	make -s clean

recompile:
	make -s clean
	make -s compile
