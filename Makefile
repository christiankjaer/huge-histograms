nvflags=-O3 -I../cub-1.5.2 -std=c++11 -Wno-deprecated-gpu-targets --gpu-architecture=compute_35 --gpu-code=compute_35
nvcom=nvcc $(nvflags)

histtest: NaiveHistTest.cu
	$(nvcom) -o histtest NaiveHistTest.cu

stream: StreamHistTest.cu
	$(nvcom) -o stream StreamHistTest.cu

clean:
	rm -f histtest
	rm -f stream

recompile:
	make -s clean
	make -s histtest
	make -s stream
