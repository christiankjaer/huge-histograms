default:
	nvcc -Wno-deprecated-gpu-targets -o hist ./histMain.cu	

compile:
	nvcc -Wno-deprecated-gpu-targets -o radix ./radix.cu

clean:
	rm -f ./radix
	rm -f ./hist

radix:
	make -s recompile
	./radix
	make -s clean

recompile:
	make -s clean
	make -s compile
