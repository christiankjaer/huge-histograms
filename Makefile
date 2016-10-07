default:
	nvcc -Wno-deprecated-gpu-targets -o hist ./histMain.cu	

compile:
	nvcc -Wno-deprecated-gpu-targets -o radix ./radix.cu
	nvcc -Wno-deprecated-gpu-targets -o tests ./tests.cu

clean:
	rm -f ./radix
	rm -f ./hist
	rm -f ./tests

tests:
	make -s recompile
	./tests
	make -s clean

radix:
	make -s recompile
	./radix
	make -s clean

recompile:
	make -s clean
	make -s compile
