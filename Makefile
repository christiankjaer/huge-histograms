nvflags=-O3 -I../cub-1.5.2 -std=c++11 -Wno-deprecated-gpu-targets --gpu-architecture=compute_35 --gpu-code=compute_35
nvcom=nvcc $(nvflags)

default:
	$(nvcom) -o hist ./histMain.cu

asyncTest: asyncHistTest.cu
	$(nvcom) -o asyncTest asyncHistTest.cu

histtest: NaiveHistTest.cu
	$(nvcom) -o histtest NaiveHistTest.cu

clean:
	rm -f ./hist
	rm -f ./tests
	rm -f ./segind
	rm -f ./histtest
	rm -f ./asyncTest

segind:
	make -s recompile
	./segind
	make -s clean

memalloc:
	$(nvcom) -o malloc ./CUDAMALLOC.cu
	./malloc
	rm -rf ./malloc

tests:
	make -s clean
	$(nvcom) -o tests  ./tests.cu
	./tests
	make -s clean

recompile:
	make -s clean
	make -s compile
