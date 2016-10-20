default:
	nvcc -Wno-deprecated-gpu-targets -o hist ./histMain.cu

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

tests:
	make -s clean
	nvcc -Wno-deprecated-gpu-targets -o tests  ./tests.cu
	./tests
	make -s clean

recompile:
	make -s clean
	make -s compile

backup:
	make -s clean
	cp -r ./* ../backup/
