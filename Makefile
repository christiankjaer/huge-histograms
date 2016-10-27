nvflags=-I../cub-1.5.2 -std=c++11 -Wno-deprecated-gpu-targets
nvcom=nvcc $(nvflags)

default:
	$(nvcom) -o hist ./histMain.cu

compile:
#nvcc -Wno-deprecated-gpu-targets -o segind ./SegmentIndex.cu

clean:
	rm -f ./hist
	rm -f ./tests
	rm -f ./segind

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

backup:
	make -s clean
	cp -r ./* ../backup/
