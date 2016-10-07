
compile:
	nvcc -Wno-deprecated-gpu-targets -o radix ./radix.cu

clean:
	rm -f ./radix

radix:
	make -s recompile
	./radix

recompile:
	make -s clean
	make -s compile
