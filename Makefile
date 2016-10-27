nvflags=-I../cub-1.5.2
nvcom=nvcc $(nvflags)
default:
	$(nvcom) -Wno-deprecated-gpu-targets -o hist ./histMain.cu

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
	nvcc -Wno-deprecated-gpu-targets -o malloc ./CUDAMALLOC.cu
	./malloc
	rm -rf ./malloc

tests:
	make -s clean
	$(nvcom) -Wno-deprecated-gpu-targets -o tests  ./tests.cu
	./tests
	make -s clean

recompile:
	make -s clean
	make -s compile

backup:
	make -s clean
	cp -r ./* ../backup/
